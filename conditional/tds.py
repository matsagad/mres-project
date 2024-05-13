from conditional.wrapper import ConditionalWrapper
from model.diffusion import FrameDiffusionModel
from protein.frames import Frames
import torch
from torch import Tensor
from tqdm import tqdm
from typing import Callable
from utils.resampling import residual_resample


class TDS(ConditionalWrapper):

    def __init__(self, model: FrameDiffusionModel) -> None:
        super().__init__(model)
        self.with_config()

    def with_config(
        self,
        resample_indices: Callable = residual_resample,
    ) -> "TDS":
        self.resample_indices = resample_indices
        return self

    def sample_given_motif(
        self, mask: Tensor, motif: Tensor, motif_mask: Tensor
    ) -> Tensor:
        """
        Twisted Diffusion Sampler (for fixed motif inpainting) as defined
        in the paper: https://arxiv.org/pdf/2306.17775 (Wu et al., 2023)
        """

        N_TIMESTEPS = self.model.n_timesteps
        K = mask.shape[0]

        # Set-up motif
        x_motif = self.model.coords_to_frames(motif, motif_mask)

        # Initialise weights
        T = torch.tensor([N_TIMESTEPS - 1] * K, device=self.device).long()
        x_T = self.model.sample_frames(mask)
        log_p_tilde_T = self.optimal_twisting_log_likelihood(
            x_motif, x_T, T, motif_mask, mask
        )
        log_p_tilde_T -= torch.logsumexp(log_p_tilde_T, dim=0)

        w = torch.exp(log_p_tilde_T)
        w /= torch.sum(w)

        ess = (1 / torch.sum(w**2)).cpu()

        x_t = x_T
        log_p_tilde_t = log_p_tilde_T

        x_trajectory = [x_T]
        pf_stats = {"ess": [], "w": []}

        with torch.no_grad():
            for i in tqdm(
                reversed(range(N_TIMESTEPS)),
                desc="Reverse diffusing samples",
                total=N_TIMESTEPS,
                disable=not self.verbose,
            ):
                # Resample particles
                if ess < 0.5 * K:
                    resampled_indices = self.resample_indices(w)
                    x_t.rots = x_t.rots[resampled_indices]
                    x_t.trans = x_t.trans[resampled_indices]
                    log_p_tilde_t = log_p_tilde_t[resampled_indices]

                t = torch.tensor([i] * K, device=self.device).long()

                # Conditional score approximation s_{\theta}(x^{t + 1}, y=x_{M}^{0})
                ## [B, N_AA, 3]  +  [B, 1, 1]
                cond_score = self.model.score(
                    x_t, t, mask
                ) + self.optimal_twisting_grad_log_likelihood(
                    x_motif, x_t, t, motif_mask, mask
                ).view(
                    -1, 1, 1
                )

                # Propose next step
                x_t_plus_one = x_t
                t_plus_one = t

                x_t = self.reverse_diffuse_with_conditional_score(
                    x_t_plus_one, t_plus_one, mask, cond_score
                )
                t = t_plus_one - 1
                x_trajectory.append(x_t)

                # Update weights
                ## Note: we normalise each component here to avoid (possibly rare?) overflows
                ## but this shouldn't change the final quantity
                log_p_tilde_t_plus_one = log_p_tilde_t

                log_p_tilde_t = self.optimal_twisting_log_likelihood(
                    x_motif, x_t, t, motif_mask, mask
                )
                log_p_tilde_t -= torch.logsumexp(log_p_tilde_t, dim=0)

                reverse_llik = self.model.reverse_log_likelihood(
                    x_t, x_t_plus_one, t_plus_one, mask, mask
                )
                reverse_llik -= torch.logsumexp(reverse_llik, dim=0)

                reverse_cond_llik = self.reverse_log_likelihood_with_conditional_score(
                    x_t, x_t_plus_one, t_plus_one, mask, cond_score
                )
                reverse_cond_llik -= torch.logsumexp(reverse_cond_llik, dim=0)

                log_w = (reverse_llik + log_p_tilde_t) - (
                    reverse_cond_llik + log_p_tilde_t_plus_one
                )
                log_w = log_w - torch.max(log_w)
                log_sum_w = torch.logsumexp(log_w, dim=0)
                w = torch.exp(log_w - log_sum_w)
                w /= torch.sum(w)

                ess = (1 / torch.sum(w**2)).cpu()

                pf_stats["w"].append(w.cpu())
                pf_stats["ess"].append(ess)

        self.save_stats(pf_stats)

        return x_trajectory

    def optimal_twisting_log_likelihood(
        self, x_motif: Frames, x_t: Frames, t: Tensor, motif_mask: Tensor, mask: Tensor
    ) -> Tensor:
        # \log(\tilde{p}_{\theta}(y | x^{t}, M))
        MOTIF_SEGMENT = motif_mask[0] == 1

        # TODO: Is it correct to center the motif region at zero? Otherwise predicted
        # \hat{x}^{0} can be magnitudes away.
        mu = self.model.predict_fully_denoised(x_t, t, mask).trans[:, MOTIF_SEGMENT]
        mu = mu - torch.mean(mu, dim=1, keepdim=True)
        sigma_squared = self.model.forward_variance[t]

        log_density = (
            -0.5
            * ((x_motif.trans[:, MOTIF_SEGMENT] - mu) ** 2)
            / sigma_squared.view(-1, 1, 1)
        ).sum(
            axis=(1, 2)
        )  # - 0.5 * torch.log(2 * torch.pi * sigma_squared)

        return log_density

    def optimal_twisting_grad_log_likelihood(
        self, x_motif: Frames, x_t: Frames, t: Tensor, motif_mask: Tensor, mask: Tensor
    ) -> Tensor:
        # \nabla_{x^{t}}\log(\tilde{p}_{\theta}(y | x^{t}, M))
        MOTIF_SEGMENT = motif_mask[0] == 1

        mu = self.model.predict_fully_denoised(x_t, t, mask).trans[:, MOTIF_SEGMENT]
        mu = mu - torch.mean(mu, dim=1, keepdim=True)
        sigma_squared = self.model.forward_variance[t].view(-1, 1, 1)

        grad_log_density = (
            -(x_motif.trans[:, MOTIF_SEGMENT] - mu) / sigma_squared
        ).sum(axis=(1, 2))

        return grad_log_density

    def reverse_diffuse_with_conditional_score(
        self,
        x_t: Frames,
        t: Tensor,
        mask: Tensor,
        score: Tensor,
        noise_scale: float = 1,
    ) -> Frames:
        # \sim \tilde{p}_{\theta}(x^{t - 1} | x^{t}, y)
        mu = (x_t.trans + self.model.variance[t].view(-1, 1, 1) * score) / torch.sqrt(
            1 - self.model.variance[t].view(-1, 1, 1)
        )
        sigma_tilde = self.model.sqrt_variance[t].view(-1, 1, 1)

        sample_trans = mu + noise_scale * sigma_tilde * torch.randn(
            x_t.trans.shape, device=self.device
        )
        sample_trans *= mask[..., None]
        x_t_minus_one = self.model.coords_to_frames(sample_trans, mask)

        return x_t_minus_one

    def reverse_log_likelihood_with_conditional_score(
        self,
        x_t_minus_one: Frames,
        x_t: Frames,
        t: Tensor,
        mask: Tensor,
        score: Tensor,
    ) -> Tensor:
        # \log(\tilde{p}_{\theta}(x^{t - 1} | x^{t}, y))
        # We assume mask is the same for all the particles.
        MASK_SEGMENT = mask[0] == 1

        mu = (
            x_t.trans[:, MASK_SEGMENT] + self.model.variance[t].view(-1, 1, 1) * score
        ) / torch.sqrt(1 - self.model.variance[t].view(-1, 1, 1))
        sigma_tilde_squared = self.model.variance[t]

        log_density = (
            -0.5
            * ((x_t_minus_one.trans[:, MASK_SEGMENT] - mu) ** 2)
            / sigma_tilde_squared.view(-1, 1, 1)
        ).sum(
            axis=(1, 2)
        )  # - 0.5 * torch.log(2 * torch.pi * sigma_tilde_squared)

        return log_density
