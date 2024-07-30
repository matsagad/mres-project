from conditional import register_conditional_method
from conditional.components.particle_filter import ParticleFilter
from conditional.wrapper import ConditionalWrapper, ConditionalWrapperConfig
import logging
from model.diffusion import FrameDiffusionModel
import torch
from torch import Tensor
from utils.resampling import get_resampling_method
from tqdm import tqdm

logger = logging.getLogger(__name__)


class SMCDiffConfig(ConditionalWrapperConfig):
    n_batches: int
    noisy_motif: bool
    particle_filter: bool
    replacement_weight: float
    resampling_method: str


@register_conditional_method("smcdiff", SMCDiffConfig)
class SMCDiff(ConditionalWrapper, ParticleFilter):

    def __init__(self, model: FrameDiffusionModel) -> None:
        super().__init__(model)
        self.with_config()

        self.supports_condition_on_motif = True
        self.supports_condition_on_symmetry = False

        self.model.compute_unique_only = True

    def with_config(
        self,
        n_batches: int = 1,
        noisy_motif: bool = False,
        particle_filter: bool = False,
        replacement_weight: float = 1.0,
        resampling_method: str = "residual",
    ) -> "SMCDiff":
        self.n_batches = n_batches
        self.noisy_motif = noisy_motif
        self.particle_filter = particle_filter
        self.replacement_weight = replacement_weight

        # We don't resample when particle_filter=False (i.e. for replacement method)
        self.resample_indices = (
            None
            if resampling_method is None
            else get_resampling_method(resampling_method)
        )
        return self

    def sample_given_motif(
        self,
        mask: Tensor,
        motif: Tensor,
        motif_mask: Tensor,
    ) -> Tensor:
        """
        Replacement method as defined in the ProtDiff/SMCDiff
        paper: https://arxiv.org/pdf/2206.04119.pdf (Trippe et al., 2023)
        """
        N_BATCHES = self.n_batches
        N_TIMESTEPS = self.model.n_timesteps
        K = mask.shape[0]
        N_MOTIFS = motif_mask.shape[0]
        K_batch = K // N_BATCHES
        assert (
            K % N_BATCHES == 0
        ), f"Number of batches {N_BATCHES} does not divide number of particles {K}"

        # (1) Forward diffuse motif
        OBSERVED_REGION = torch.sum(motif_mask, dim=0) == 1
        x_motif = self.model.coords_to_frames(motif, motif_mask)

        motif_trajectory = [x_motif]
        forward_diffuse = (
            self.model.forward_diffuse
            if self.noisy_motif
            else self.model.forward_diffuse_deterministic
        )

        for i in tqdm(
            range(N_TIMESTEPS),
            desc="Forward diffusing motif",
            total=N_TIMESTEPS,
            disable=not self.verbose,
        ):
            t = torch.tensor([i] * motif_mask.shape[0], device=self.device).long()
            x_motif = forward_diffuse(x_motif, t, motif_mask)
            # Keep motif zero-centred
            for j in range(N_MOTIFS):
                x_motif.trans[j, motif_mask[j] == 1] -= torch.mean(
                    x_motif.trans[j, motif_mask[j] == 1], dim=0, keepdim=True
                )
            motif_trajectory.append(x_motif)

        logger.info("Collected noised motifs at all time steps.")

        # (2) Reverse diffuse particles
        x_T = self.model.sample_frames(mask)
        x_trajectory = [x_T]
        x_t = x_T
        gamma = self.replacement_weight

        if self.particle_filter:
            pf_stats = {"ess": [], "w": []}
            w = torch.ones((N_BATCHES, K_batch), device=self.device) / K_batch
            ess = torch.zeros(N_BATCHES, device=self.device)

        T = torch.tensor([N_TIMESTEPS - 1] * K, device=self.device).long()
        score_T = self.model.score(x_T, T, mask)
        score_t = score_T

        with torch.no_grad():
            for i in tqdm(
                reversed(range(N_TIMESTEPS)),
                desc="Reverse diffusing samples",
                total=N_TIMESTEPS,
                disable=not self.verbose,
            ):
                t = torch.tensor([i] * K, device=self.device).long()

                # Keep motif segment of protein zero-centred for visuals
                x_t.trans[:, mask[0] == 1] -= torch.mean(
                    x_t.trans[:, OBSERVED_REGION], dim=1
                ).unsqueeze(1)

                # Replace motif
                ## Index by i + 1 since ts_motifs[(T - 1) + 1] is x_{M}^{T}
                for j in range(N_MOTIFS):
                    ## Position motif to the same centre of mass as the motif region
                    com_offset = torch.mean(
                        x_t.trans[:, motif_mask[j] == 1], dim=1
                    ).unsqueeze(1)

                    x_t.trans[:, motif_mask[j] == 1] = (1 - gamma) * (
                        x_t.trans[:, motif_mask[j] == 1]
                    ) + gamma * (
                        motif_trajectory[t[0] + 1].trans[j : j + 1, motif_mask[j] == 1]
                        + com_offset
                    )
                x_t = self.model.coords_to_frames(x_t.trans, mask)

                if not self.particle_filter:
                    x_t = self.model.reverse_diffuse(x_t, t, mask)
                    x_trajectory.append(x_t)
                    continue

                # Re-weight based on motif at t-1
                ## Find likelihood of getting motif when de-noised
                log_w = 0
                with self.model.with_score(score_t):
                    for j in range(N_MOTIFS):
                        x_motif_j = self.model.coords_to_frames(
                            motif_trajectory[i].trans[j : j + 1],
                            motif_mask[j : j + 1],
                        )
                        log_w += self.model.reverse_log_likelihood(
                            x_motif_j,
                            x_t,
                            t,
                            motif_mask,
                            mask,
                        ).view(N_BATCHES, K_batch)
                log_sum_w = torch.logsumexp(log_w, dim=1, keepdim=True)

                w *= torch.exp(log_w - log_sum_w)
                all_zeros = torch.all(w == 0, dim=1)

                w[all_zeros] = 1 / K_batch
                ess[all_zeros] = 0
                w[~all_zeros] /= w[~all_zeros].sum(dim=1, keepdim=True)
                ess[~all_zeros] = 1 / torch.sum(w[~all_zeros] ** 2, dim=1)

                ### Collect particle filtering stats
                pf_stats["ess"].append(ess.cpu())
                pf_stats["w"].append(w.cpu())

                # Resample particles
                self.resample(w, ess, [x_t.rots, x_t.trans])

                # Propose next step
                score_t = self.model.score(x_t, t, mask)
                with self.model.with_score(score_t):
                    x_t = self.model.reverse_diffuse(x_t, t, mask)
                x_trajectory.append(x_t)

        # Save traces for debugging
        if self.particle_filter:
            self.save_stats(pf_stats)

        logger.info("Done de-noising samples.")

        return x_trajectory

    def sample_given_symmetry(self, mask: Tensor, symmetry: str) -> Tensor:
        raise NotImplementedError("")
