from conditional.wrapper import ConditionalWrapper
import logging
from model.diffusion import FrameDiffusionModel
import torch
from torch import Tensor
from utils.resampling import residual_resample
from tqdm import tqdm
from typing import Callable

logger = logging.getLogger(__name__)


class ReplacementMethod(ConditionalWrapper):

    def __init__(self, model: FrameDiffusionModel) -> None:
        super().__init__(model)
        self.with_config()

    def with_config(
        self,
        noisy_motif: bool = False,
        particle_filter: bool = False,
        replacement_weight: float = 1.0,
        resample_indices: Callable = residual_resample,
    ) -> "ReplacementMethod":
        self.noisy_motif = noisy_motif
        self.particle_filter = particle_filter
        self.replacement_weight = replacement_weight
        self.resample_indices = resample_indices
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

        NOISE_SCALE = self.model.noise_scale
        N_TIMESTEPS = self.model.n_timesteps
        K = mask.shape[0]

        # (1) Forward diffuse motif
        MOTIF_SEGMENT = motif_mask[0] == 1
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
            motif_trajectory.append(x_motif)

        logger.info("Collected noised motifs at all time steps.")

        # (2) Reverse diffuse particles
        x_T = self.model.sample_frames(mask)
        x_trajectory = [x_T]
        x_t = x_T
        _gamma = self.replacement_weight

        if self.particle_filter:
            pf_stats = {"ess": [], "w": []}
            w = torch.ones(K, device=self.device) / K

        with torch.no_grad():
            for i in tqdm(
                reversed(range(N_TIMESTEPS)),
                desc="Reverse diffusing samples",
                total=N_TIMESTEPS,
                disable=not self.verbose,
            ):
                # Replace motif
                ## Index by i + 1 since ts_motifs[(T - 1) + 1] is x_{M}^{T}
                x_t.trans[:, MOTIF_SEGMENT] = (1 - _gamma) * x_t.trans[
                    :, MOTIF_SEGMENT
                ] + _gamma * motif_trajectory[i + 1].trans[:, MOTIF_SEGMENT]
                x_t = self.model.coords_to_frames(x_t.trans, mask)

                t = torch.tensor([i] * K, device=self.device).long()

                if not self.particle_filter:
                    x_t = self.model.reverse_diffuse(
                        x_t, t, mask, noise_scale=NOISE_SCALE
                    )
                    x_trajectory.append(x_t)
                    continue

                # Re-weight based on motif at t-1
                ## Find likelihood of getting motif when de-noised
                log_w = self.model.reverse_log_likelihood(
                    motif_trajectory[i], x_t, t, motif_mask, mask
                )
                log_sum_w = torch.logsumexp(log_w, dim=0)
                w *= torch.exp(log_w - log_sum_w)
                w /= torch.sum(w)
                ess = (1 / torch.sum(w**2)).cpu()

                ### Collect particle filtering stats
                pf_stats["ess"].append(ess)
                pf_stats["w"].append(w.cpu())

                # Resample particles
                if ess < 0.5 * K:
                    resampled_indices = self.resample_indices(w)
                    x_t.rots = x_t.rots[resampled_indices]
                    x_t.trans = x_t.trans[resampled_indices]
                    w = torch.ones(K, device=self.device) / K

                # Propose next step
                x_t = self.model.reverse_diffuse(x_t, t, mask, noise_scale=NOISE_SCALE)
                x_trajectory.append(x_t)

        # Save traces for debugging
        if self.particle_filter:
            self.save_stats(pf_stats)

        logger.info("Done de-noising samples.")

        return x_trajectory
