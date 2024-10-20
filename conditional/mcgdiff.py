from conditional import register_conditional_method
from conditional.components.particle_filter import ParticleFilter, LikelihoodMethod
from conditional.wrapper import ConditionalWrapper, ConditionalWrapperConfig
import logging
from model.diffusion import FrameDiffusionModel
import torch
from torch import Tensor
from utils.resampling import get_resampling_method
from tqdm import tqdm

logger = logging.getLogger(__name__)


class MCGDiffConfig(ConditionalWrapperConfig):
    n_batches: int
    particle_filter: bool
    resampling_method: str


@register_conditional_method("mcgdiff", MCGDiffConfig)
class MCGDiff(ConditionalWrapper, ParticleFilter):

    def __init__(self, model: FrameDiffusionModel) -> None:
        super().__init__(model)
        self.with_config()

        self.supports_condition_on_motif = True
        self.supports_condition_on_symmetry = False

        self.model.compute_unique_only = False

    def with_config(
        self,
        n_batches: int = 1,
        particle_filter: bool = False,
        resampling_method: str = "residual",
    ) -> "MCGDiff":
        self.n_batches = n_batches
        self.particle_filter = particle_filter

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
        Monte Carlo guidden Diffusion (for noiseless inpainting) as in
        the paper: https://arxiv.org/pdf/2308.07983 (Cardoso et al., 2023)

        (We change notation: xi -> x, K -> gamma)
        """
        N_BATCHES = self.n_batches
        N_TIMESTEPS = self.model.n_timesteps
        K = mask.shape[0]

        K_batch = K // N_BATCHES
        assert (
            K % N_BATCHES == 0
        ), f"Number of batches {N_BATCHES} does not divide number of particles {K}"

        OBSERVED_REGION = torch.sum(motif_mask, dim=0) == 1
        y = self.model.coords_to_frames(motif, motif_mask)
        y_mask = motif_mask

        # Reverse diffuse particles
        T = torch.tensor([N_TIMESTEPS - 1] * K, device=self.device).long()
        alpha_bar_T = 1 - self.model.forward_variance[T[0]]
        gamma_T = 0.5

        x_T = self.model.sample_frames(mask)
        x_T.trans[:, OBSERVED_REGION] = gamma_T * (
            torch.sqrt(alpha_bar_T) * y.trans[:, OBSERVED_REGION]
            + (1 - alpha_bar_T) * x_T.trans[:, OBSERVED_REGION]
        )
        x_T = self.model.coords_to_frames(x_T.trans, mask)

        x_trajectory = [x_T]
        x_t = x_T
        x_t.trans[:, mask[0] == 1] -= torch.mean(
            x_t.trans[:, OBSERVED_REGION], dim=1
        ).unsqueeze(1)

        pf_stats = {"ess": [], "w": []}
        w = torch.ones((N_BATCHES, K_batch), device=self.device) / K_batch
        ess = torch.zeros(N_BATCHES, device=self.device)

        log_likelihood = self.get_log_likelihood(LikelihoodMethod.FRAME_MASK)

        for i in tqdm(
            reversed(range(N_TIMESTEPS - 1)),
            desc="Reverse diffusing samples",
            total=N_TIMESTEPS,
            initial=1,
            disable=not self.verbose,
        ):
            t = torch.tensor([i] * K, device=self.device).long()

            alpha_bar_t = 1 - self.model.forward_variance[t[0]]
            alpha_bar_t_plus_one = 1 - self.model.forward_variance[t[0] + 1]
            sigma_squared_t_plus_one = self.model.variance[t[0] + 1]
            sigma_t_plus_one = self.model.sqrt_variance[t[0] + 1]
            y_t_plus_one = self.model.coords_to_frames(
                torch.sqrt(alpha_bar_t_plus_one) * y.trans, y_mask
            )
            gamma_t = self.model.variance[t[0] + 1] / (
                self.model.variance[t[0] + 1] + self.model.variance[t[0]]
            )

            y_t = self.model.coords_to_frames(torch.sqrt(alpha_bar_t) * y.trans, y_mask)

            score_t = self.model.score(x_t, t, mask)
            with self.model.with_score(score_t):
                mu_t_minus_one = self.model.reverse_diffuse_deterministic(x_t, t, mask)
                mu_t_minus_one.trans[:, mask[0] == 1] -= torch.mean(
                    mu_t_minus_one.trans[:, OBSERVED_REGION], dim=1
                ).unsqueeze(1)

            # Compute weights
            if i == N_TIMESTEPS - 2:
                log_w = log_likelihood(mu_t_minus_one, y_t, y_mask, 2 - alpha_bar_t)
            else:
                log_w = log_likelihood(
                    mu_t_minus_one,
                    y_t,
                    y_mask,
                    sigma_squared_t_plus_one + 1 - alpha_bar_t,
                ) - log_likelihood(x_t, y_t_plus_one, y_mask, 1 - alpha_bar_t_plus_one)
            log_w = log_w.view(N_BATCHES, K_batch)
            log_sum_w = torch.logsumexp(log_w, dim=1, keepdim=True)

            w *= torch.exp(log_w - log_sum_w)
            all_zeros = torch.all(w == 0, dim=1)

            w[all_zeros] = 1 / K_batch
            ess[all_zeros] = 0
            w[~all_zeros] /= w[~all_zeros].sum(dim=1, keepdim=True)
            ess[~all_zeros] = 1 / torch.sum(w[~all_zeros] ** 2, dim=1)

            ## Collect particle filtering stats
            pf_stats["ess"].append(ess.cpu())
            pf_stats["w"].append(w.cpu())

            # Resample particles
            self.resample(w, ess, [mu_t_minus_one.trans])

            # Propose next step
            z_t = torch.randn(x_t.trans.shape, device=self.device)
            x_t_minus_one_trans = torch.zeros(
                mu_t_minus_one.trans.shape, device=self.device
            )
            x_t_minus_one_trans[:, ~OBSERVED_REGION] = (
                mu_t_minus_one.trans[:, ~OBSERVED_REGION]
                + sigma_t_plus_one * z_t[:, ~OBSERVED_REGION]
            )
            x_t_minus_one_trans[:, OBSERVED_REGION] = (
                gamma_t * y_t.trans[:, OBSERVED_REGION]
                + (1 - gamma_t) * mu_t_minus_one.trans[:, OBSERVED_REGION]
                + self.model.sqrt_variance[t[0]]
                * torch.sqrt(gamma_t)
                * z_t[:, OBSERVED_REGION]
            )
            x_t = self.model.coords_to_frames(x_t_minus_one_trans, mask)
            x_trajectory.append(x_t)

        # Save traces for debugging
        if self.particle_filter:
            self.save_stats(pf_stats)

        logger.info("Done de-noising samples.")

        return x_trajectory

    def sample_given_symmetry(self, mask: Tensor, symmetry: str) -> Tensor:
        raise NotImplementedError("")

    def sample_given_motif_and_symmetry(
        self, mask: Tensor, motif: Tensor, motif_mask: Tensor, symmetry: str
    ) -> Tensor:
        raise NotImplementedError("")
