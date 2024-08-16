from conditional import register_conditional_method
from conditional.components.observation_generator import (
    LinearObservationGenerator,
    ObservationGenerationMethod,
)
from conditional.components.particle_filter import (
    ParticleFilter,
    LikelihoodMethod,
    LikelihoodReduction,
)
from conditional.wrapper import ConditionalWrapper, ConditionalWrapperConfig
from functools import partial
from enum import Enum
from model.diffusion import FrameDiffusionModel
import torch
from torch import Tensor
from tqdm import tqdm
from typing import Callable, List
from utils.resampling import get_resampling_method


class BPFMethod(str, Enum):
    NOISED_TARGETS = "noised_targets"
    PROJECTION = "projection"


class BPFConfig(ConditionalWrapperConfig):
    n_batches: int
    conditional_method: BPFMethod
    fixed_motif: bool
    observed_sequence_method: ObservationGenerationMethod
    observed_sequence_noised: bool
    likelihood_method: LikelihoodMethod
    likelihood_sigma: float
    rot_likelihood_scale: float
    particle_filter: bool
    resampling_method: str


@register_conditional_method("bpf", BPFConfig)
class BPF(ConditionalWrapper, ParticleFilter, LinearObservationGenerator):

    def __init__(self, model: FrameDiffusionModel) -> None:
        super().__init__(model)
        self.with_config()

        self.supports_condition_on_motif = True
        self.supports_condition_on_symmetry = False

        # Supported only for noised targets conditional method
        self.model.compute_unique_only = True

    def with_config(
        self,
        n_batches: int = 1,
        conditional_method: BPFMethod = BPFMethod.PROJECTION,
        fixed_motif: bool = True,
        observed_sequence_method: ObservationGenerationMethod = ObservationGenerationMethod.BACKWARD,
        observed_sequence_noised: bool = True,
        likelihood_method: LikelihoodMethod = LikelihoodMethod.MASK,
        likelihood_sigma: float = 0.05,
        rot_likelihood_scale: float = 64.0,
        particle_filter: bool = True,
        resampling_method: str = "residual",
    ) -> "BPF":
        self.n_batches = n_batches
        self.conditional_method = conditional_method
        self.fixed_motif = fixed_motif
        self.observed_sequence_method = observed_sequence_method
        self.observed_sequence_noised = observed_sequence_noised
        self.likelihood_method = likelihood_method
        self.likelihood_sigma = likelihood_sigma
        self.rot_likelihood_scale = rot_likelihood_scale
        self.particle_filter = particle_filter
        self.resample_indices = get_resampling_method(resampling_method)

        self.model.compute_unique_only = conditional_method != BPFMethod.PROJECTION
        return self

    def sample_given_motif(
        self, mask: Tensor, motif: Tensor, motif_mask: Tensor
    ) -> Tensor:
        N_RESIDUES = (mask[0] == 1).sum().item()
        N_COORDS_PER_RESIDUE = 3
        D = N_RESIDUES * N_COORDS_PER_RESIDUE

        A = []
        if self.conditional_method == BPFMethod.NOISED_TARGETS:
            for i in range(len(motif_mask)):
                n_motif_residues = (motif_mask[i] == 1).sum().item()
                d = n_motif_residues * N_COORDS_PER_RESIDUE

                motif_indices_flat = torch.where(
                    torch.repeat_interleave(
                        motif_mask[i, :N_RESIDUES] == 1, N_COORDS_PER_RESIDUE
                    )
                )[0]
                A_motif = torch.zeros((d, D), device=self.device)
                A_motif[range(d), motif_indices_flat] = 1
                A.append(A_motif)

        log_likelihood = self.get_log_likelihood(self.likelihood_method)

        if self.likelihood_method == LikelihoodMethod.MATRIX:
            log_likelihood = partial(log_likelihood, A=A)

        if not self.fixed_motif:
            log_likelihood = partial(log_likelihood, reduce=LikelihoodReduction.SUM)
        if self.likelihood_method == LikelihoodMethod.FRAME_BASED_DISTANCE:
            log_likelihood = partial(
                log_likelihood, rot_likelihood_scale=self.rot_likelihood_scale
            )

        return self.sample_conditional(
            mask, motif, motif_mask, A, log_likelihood, recenter_y=True, recenter_x=True
        )

    def sample_given_symmetry(self, mask: Tensor, symmetry: str) -> Tensor:
        A, y, y_mask = self._get_symmetric_constraints(mask, symmetry)

        assert self.likelihood_method == LikelihoodMethod.MATRIX, (
            f"Likelihood method '{self.likelihood_method}' is not supported for sampling symmetry."
            f" Make sure to set experiment.conditional_method.likelihood_method='{LikelihoodMethod.MATRIX}'"
        )
        log_likelihood = partial(self.get_log_likelihood(self.likelihood_method), A=[A])

        return self.sample_conditional(
            mask, y, y_mask, [A], log_likelihood, recenter_y=False, recenter_x=True
        )

    def sample_conditional(
        self,
        mask: Tensor,
        y: Tensor,
        y_mask: Tensor,
        A: List[Tensor],
        log_likelihood: Callable,
        recenter_y: bool = True,
        recenter_x: bool = True,
    ) -> Tensor:
        """
        Bootstrap Particle Filter
        """
        N_COORDS_PER_RESIDUE = 3
        N_BATCHES = self.n_batches
        N_TIMESTEPS = self.model.n_timesteps
        N_MOTIFS = y_mask.shape[0]
        K, MAX_N_RESIDUES = mask.shape
        K_batch = K // N_BATCHES
        assert (
            K % N_BATCHES == 0
        ), f"Number of batches {N_BATCHES} does not divide number of particles {K}"

        OBSERVED_REGION = torch.sum(y_mask, dim=0) == 1
        N_RESIDUES = (mask[0] == 1).sum().item()

        # (1) Setup likelihood measure and sequence {y_t}
        y_zero = self.model.coords_to_frames(y, y_mask)
        for j in range(N_MOTIFS):
            y_zero.trans[j, y_mask[j] == 1] -= torch.mean(
                y_zero.trans[j, y_mask[j] == 1], dim=0, keepdim=True
            )

        if self.conditional_method == BPFMethod.NOISED_TARGETS:
            epsilon = torch.randn(
                (N_TIMESTEPS, N_RESIDUES * N_COORDS_PER_RESIDUE), device=self.device
            )
            if self.observed_sequence_method == ObservationGenerationMethod.BACKWARD:
                # Use noise-sharing for backwards generation case
                _x_T_trans = torch.zeros(
                    (K, MAX_N_RESIDUES, N_COORDS_PER_RESIDUE), device=self.device
                )
                _x_T_trans[:, :N_RESIDUES] = epsilon[-1].view(
                    1, N_RESIDUES, N_COORDS_PER_RESIDUE
                )
                x_T = self.model.coords_to_frames(_x_T_trans, mask)
            else:
                x_T = self.model.sample_frames(mask)
            y_sequence = self.generate_observed_sequence(
                mask, y_zero, y_mask, A, epsilon, recenter_y=recenter_y
            )
        else:
            x_T = self.model.sample_frames(mask)
            y_sequence = []

        # (2) Generate sequence {x_t}
        x_sequence = [x_T]
        x_t = x_T

        w = torch.ones((N_BATCHES, K_batch), device=self.device) / K_batch
        ess = torch.zeros(N_BATCHES, device=self.device)
        pf_stats = {"ess": [], "w": []}
        FINAL_TIME_STEP = 0

        with torch.no_grad():
            ## Compute score to be used in the first iteration
            T = torch.tensor([N_TIMESTEPS - 1] * K, device=self.device).long()
            score_T = self.model.score(x_T, T, mask)
            score_t = score_T

            for i in tqdm(
                reversed(range(N_TIMESTEPS)),
                desc="Generating {x_t}",
                total=N_TIMESTEPS,
                disable=not self.verbose,
            ):
                t = torch.tensor([i] * K, device=self.device).long()

                # Propose next step
                with self.model.with_score(score_t):
                    x_t_minus_one = self.model.reverse_diffuse(x_t, t, mask)

                ## Translate mean so that motif segment is centred at zero
                if recenter_x:
                    x_t_minus_one.trans[:, :N_RESIDUES] -= torch.mean(
                        x_t_minus_one.trans[:, OBSERVED_REGION], dim=1
                    ).unsqueeze(1)
                x_sequence.append(x_t_minus_one)
                if i == FINAL_TIME_STEP:
                    continue

                ## Compute score to be used in next iteration
                t_minus_one = t - 1
                # score_t_minus_one = self.model.score(x_t_minus_one, t_minus_one, mask)

                if not self.particle_filter:
                    x_t = x_t_minus_one
                    # score_t = score_t_minus_one
                    continue

                # Resample particles
                ## Compute log likelihood
                alpha_bar_t = 1 - self.model.forward_variance[t[0]]
                if self.conditional_method == BPFMethod.NOISED_TARGETS:
                    variance_t = alpha_bar_t * self.likelihood_sigma**2
                    llik_x_t = x_t_minus_one
                    llik_y_t = y_sequence[t[0]]
                elif self.conditional_method == BPFMethod.PROJECTION:
                    score_t_minus_one = self.model.score(
                        x_t_minus_one, t_minus_one, mask
                    )
                    variance_t = 1 - alpha_bar_t + self.likelihood_sigma**2
                    with self.model.with_score(score_t_minus_one):
                        x_zero_hat = self.model.predict_fully_denoised(
                            x_t_minus_one, t_minus_one, mask
                        )
                    llik_x_t = x_zero_hat
                    llik_y_t = y_zero
                log_w = log_likelihood(llik_x_t, llik_y_t, y_mask, variance_t)

                log_w = log_w.view(N_BATCHES, K_batch)
                log_sum_w = torch.logsumexp(log_w, dim=1, keepdim=True)
                w *= torch.exp(log_w - log_sum_w)
                all_zeros = torch.all(w == 0, dim=1)

                w[all_zeros] = 1 / K_batch
                ess[all_zeros] = 0
                w[~all_zeros] /= w[~all_zeros].sum(dim=1, keepdim=True)
                ess[~all_zeros] = 1 / torch.sum(w[~all_zeros] ** 2, dim=1)

                pf_stats["w"].append(w.cpu())
                pf_stats["ess"].append(ess.cpu())

                if self.conditional_method == BPFMethod.NOISED_TARGETS:
                    self.resample(w, ess, [x_t_minus_one.rots, x_t_minus_one.trans])
                    score_t = self.model.score(x_t_minus_one, t_minus_one, mask)
                elif self.conditional_method == BPFMethod.PROJECTION:
                    self.resample(
                        w,
                        ess,
                        [x_t_minus_one.rots, x_t_minus_one.trans, score_t_minus_one],
                    )
                    score_t = score_t_minus_one
                x_t = x_t_minus_one

        if self.particle_filter:
            self.save_stats(pf_stats)

        if not self.fixed_motif:
            likelihoods = log_likelihood(
                llik_x_t, llik_y_t, y_mask, variance_t, reduce=LikelihoodReduction.NONE
            )
            most_likely_position = torch.argmax(likelihoods, dim=0)
            self.save_stats({"motif_mask": y_mask[most_likely_position]})

        return x_sequence
