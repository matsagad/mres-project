from conditional import register_conditional_method
from conditional.components.observation_generator import (
    LinearObservationGenerator,
    ObservationGenerationMethod,
)
from conditional.components.particle_filter import ParticleFilter, LikelihoodMethod
from conditional.wrapper import ConditionalWrapper, ConditionalWrapperConfig
from enum import Enum
from model.diffusion import FrameDiffusionModel
from protein.frames import Frames
import torch
from torch import Tensor
from tqdm import tqdm
from utils.resampling import get_resampling_method


class BPFMethod(str, Enum):
    NOISED_TARGETS = "noised_targets"
    PROJECTION = "projection"


class BPFConfig(ConditionalWrapperConfig):
    n_batches: int
    conditional_method: BPFMethod
    observed_sequence_method: ObservationGenerationMethod
    observed_sequence_noised: bool
    likelihood_method: LikelihoodMethod
    likelihood_sigma: float
    particle_filter: bool
    resampling_method: str


@register_conditional_method("bpf", BPFConfig)
class BPF(ConditionalWrapper, ParticleFilter, LinearObservationGenerator):

    def __init__(self, model: FrameDiffusionModel) -> None:
        super().__init__(model)
        self.with_config()

        self.supports_condition_on_motif = True
        self.supports_condition_on_symmetry = False

        self.model.compute_unique_only = True

    def with_config(
        self,
        n_batches: int = 1,
        conditional_method: BPFMethod = BPFMethod.PROJECTION,
        observed_sequence_method: ObservationGenerationMethod = ObservationGenerationMethod.BACKWARD,
        observed_sequence_noised: bool = True,
        likelihood_method: LikelihoodMethod = LikelihoodMethod.MASK,
        likelihood_sigma: float = 0.05,
        particle_filter: bool = True,
        resampling_method: str = "residual",
    ) -> "BPF":
        self.n_batches = n_batches
        self.conditional_method = conditional_method
        self.observed_sequence_method = observed_sequence_method
        self.observed_sequence_noised = observed_sequence_noised
        self.likelihood_method = likelihood_method
        self.likelihood_sigma = likelihood_sigma
        self.particle_filter = particle_filter
        self.resample_indices = get_resampling_method(resampling_method)
        return self

    def sample_given_motif(
        self, mask: Tensor, motif: Tensor, motif_mask: Tensor
    ) -> Tensor:
        N_RESIDUES = (mask[0] == 1).sum().item()
        N_MOTIF_RESIDUES = (motif_mask[0] == 1).sum().item()
        N_COORDS_PER_RESIDUE = 3

        D = N_RESIDUES * N_COORDS_PER_RESIDUE
        d = N_MOTIF_RESIDUES * N_COORDS_PER_RESIDUE

        motif_indices_flat = torch.where(
            torch.repeat_interleave(
                motif_mask[0][:N_RESIDUES] == 1, N_COORDS_PER_RESIDUE
            )
        )[0]
        A = torch.zeros((d, D), device=self.device)
        A[range(d), motif_indices_flat] = 1

        return self.sample_conditional(
            mask, motif, motif_mask, A, recenter_y=True, recenter_x=True
        )

    def sample_given_symmetry(self, mask: Tensor, symmetry: str) -> Tensor:
        raise NotImplementedError("")

    def sample_conditional(
        self,
        mask: Tensor,
        y: Frames,
        y_mask: Tensor,
        A: Tensor,
        recenter_y: bool = True,
        recenter_x: bool = True,
    ) -> Tensor:
        """
        Bootstrap Particle Filter
        """
        N_BATCHES = self.n_batches
        N_TIMESTEPS = self.model.n_timesteps
        N_COORDS_PER_RESIDUE = 3
        K, MAX_N_RESIDUES = mask.shape
        K_batch = K // N_BATCHES
        assert (
            K % N_BATCHES == 0
        ), f"Number of batches {N_BATCHES} does not divide number of particles {K}"

        OBSERVED_REGION = y_mask[0] == 1
        N_RESIDUES = (mask[0] == 1).sum().item()
        N_OBSERVED = (OBSERVED_REGION).sum().item()
        D = N_RESIDUES * N_COORDS_PER_RESIDUE
        d = N_OBSERVED * N_COORDS_PER_RESIDUE
        assert d == A.shape[0] and D == A.shape[1]

        # (1) Setup likelihood measure and sequence {y_t}
        y_zero = self.model.coords_to_frames(
            y.view(1, -1, N_COORDS_PER_RESIDUE), y_mask
        )
        y_zero.trans -= torch.mean(
            y_zero.trans[:, OBSERVED_REGION], dim=1, keepdim=True
        )
        if self.conditional_method == BPFMethod.NOISED_TARGETS:
            if self.observed_sequence_method == ObservationGenerationMethod.BACKWARD:
                # Use noise-sharing for backwards generation case
                epsilon_T = self.model.sample_frames(mask[:1])
                x_T = self.model.coords_to_frames(
                    torch.tile(epsilon_T.trans, (K, 1, 1)), mask
                )
            else:
                x_T = self.model.sample_frames(mask)
            y_sequence = self.generate_observed_sequence(
                mask, y_zero, y_mask, A, x_T, recenter_y=recenter_y
            )
        else:
            x_T = self.model.sample_frames(mask)
            y_sequence = []
        log_likelihood = self.get_log_likelihood(self.likelihood_method)

        w = torch.ones((N_BATCHES, K_batch), device=self.device) / K_batch
        ess = torch.zeros(N_BATCHES, device=self.device)
        pf_stats = {"ess": [], "w": []}

        # (2) Generate sequence {x_t}
        x_sequence = [x_T]
        x_t = x_T

        for i in tqdm(
            reversed(range(N_TIMESTEPS)),
            desc="Generating {x_t}",
            total=N_TIMESTEPS,
            disable=not self.verbose,
        ):
            t = torch.tensor([i] * K, device=self.device).long()

            with torch.no_grad():
                score = self.model.score(x_t, t, mask)
            with self.model.with_score(score):
                x_zero_hat = self.model.predict_fully_denoised(x_t, t, mask)
                x_bar_t = self.model.reverse_diffuse(x_t, t, mask)

            if recenter_x:
                # Translate mean so that motif segment is centred at zero
                x_bar_t.trans[:, :N_RESIDUES] -= torch.mean(
                    x_bar_t.trans[:, OBSERVED_REGION], dim=1
                ).unsqueeze(1)

            if not self.particle_filter:
                x_t = x_bar_t
                x_sequence.append(x_t)
                continue

            # Resample particles
            alpha_bar_t = 1 - self.model.forward_variance[t[0]]
            if self.conditional_method == BPFMethod.NOISED_TARGETS:
                variance_t = alpha_bar_t * self.likelihood_sigma**2
                log_w = log_likelihood(x_bar_t, y_sequence[t[0]], y_mask, variance_t)
            elif self.conditional_method == BPFMethod.PROJECTION:
                variance_t = 1 - alpha_bar_t + self.likelihood_sigma**2
                log_w = log_likelihood(x_zero_hat, y_zero, y_mask, variance_t)

            log_w = log_w.view(N_BATCHES, K_batch)
            log_sum_w = torch.logsumexp(log_w, dim=1, keepdim=True)
            w *= torch.exp(log_w - log_sum_w)
            all_zeros = torch.all(w == 0, dim=1)

            w[all_zeros] = 1 / K_batch
            ess[all_zeros] = 0
            w[~all_zeros] /= w[~all_zeros].sum(dim=1, keepdim=True)
            ess[~all_zeros] = 1 / torch.sum(w[~all_zeros] ** 2, dim=1)

            pf_stats["ess"].append(ess)
            pf_stats["w"].append(w.cpu())

            x_t = x_bar_t
            self.resample(w, ess, [x_t.rots, x_t.trans])

            x_sequence.append(x_t)

        if self.particle_filter:
            self.save_stats(pf_stats)

        return x_sequence
