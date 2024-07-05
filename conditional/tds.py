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
        sigma: float = 0.05,
        twist_scale: float = 1.0,
        likelihood_method: str = "mask",
    ) -> "TDS":
        self.resample_indices = resample_indices
        self.sigma = sigma
        self.twist_scale = twist_scale
        self.likelihood_method = likelihood_method
        return self

    def sample_given_motif(
        self, mask: Tensor, motif: Tensor, motif_mask: Tensor
    ) -> Tensor:
        """
        Twisted Diffusion Sampler (for fixed motif inpainting) as defined
        in the paper: https://arxiv.org/pdf/2306.17775 (Wu et al., 2023)
        """

        N_TIMESTEPS = self.model.n_timesteps
        K, N_RESIDUES = mask.shape
        OBSERVED_REGION = motif_mask[0] == 1
        sigma = self.sigma
        twist_scale = self.twist_scale

        # Set-up motif
        x_motif = self.model.coords_to_frames(motif, motif_mask)

        if self.likelihood_method == "mask":

            def log_likelihood(x_zero_hat: Frames, t: Tensor) -> Tensor:
                centred_x_zero_hat_trans = x_zero_hat.trans[
                    :, OBSERVED_REGION
                ] - torch.mean(x_zero_hat.trans[:, OBSERVED_REGION], dim=1).unsqueeze(1)
                return -0.5 * (
                    (
                        (centred_x_zero_hat_trans - x_motif.trans[:, OBSERVED_REGION])
                        ** 2
                    )
                    / (self.model.forward_variance[t].view(-1, 1, 1) + sigma**2)
                ).sum(dim=(1, 2))

        elif self.likelihood_method == "distance":
            N_OBSERVED = torch.sum(OBSERVED_REGION)
            _i, _j = torch.triu_indices(N_OBSERVED, N_OBSERVED, offset=1)
            dist_y = torch.cdist(
                x_motif.trans[:, OBSERVED_REGION], x_motif.trans[:, OBSERVED_REGION]
            )[:, _i, _j].view(1, (N_OBSERVED * (N_OBSERVED - 1)) // 2)

            def log_likelihood(x_zero_hat: Frames, t: Tensor) -> Tensor:
                dist_x_hat_0 = torch.cdist(
                    x_zero_hat.trans[:, OBSERVED_REGION],
                    x_zero_hat.trans[:, OBSERVED_REGION],
                )[:, _i, _j].view(-1, (N_OBSERVED * (N_OBSERVED - 1)) // 2)

                return (
                    -0.5
                    * ((dist_y - dist_x_hat_0) ** 2)
                    / (self.model.forward_variance[t].view(-1, 1) + sigma**2)
                ).sum(axis=1)

        else:
            raise KeyError(
                f"No such supported likelihood method: {self.likelihood_method}."
            )

        # Sample frames
        T = torch.tensor([N_TIMESTEPS - 1] * K, device=self.device).long()
        x_T = self.model.sample_frames(mask)
        x_T.trans.requires_grad = True

        # Initialise weights
        score = self.model.score(x_T, T, mask)
        ## Custom context manager for caching score to avoid repeated calls to model
        with self.model.with_score(score):
            x_zero_hat = self.model.predict_fully_denoised(x_T, T, mask)

        ## Calculate log likelihood and its gradient
        log_p_tilde_T = log_likelihood(x_zero_hat, T)
        (grad_log_p_tilde_T,) = torch.autograd.grad(log_p_tilde_T.sum(), x_T.trans)
        log_p_tilde_T = log_p_tilde_T - torch.logsumexp(log_p_tilde_T, dim=0)

        x_T.trans = x_T.trans.detach()

        with torch.no_grad():
            w = torch.exp(log_p_tilde_T)
            w /= torch.sum(w)
            ess = (1 / torch.sum(w**2)).cpu()

        x_t = x_T
        log_p_tilde_t = log_p_tilde_T.detach()
        grad_log_p_tilde_t = grad_log_p_tilde_T.detach()

        x_trajectory = [x_T]
        pf_stats = {"ess": [], "w": []}

        with torch.no_grad():
            for i in tqdm(
                reversed(range(N_TIMESTEPS - 1)),
                desc="Reverse diffusing samples",
                total=N_TIMESTEPS,
                disable=not self.verbose,
            ):
                # Resample particles
                if ess <= 0.5 * K:
                    resampled_indices = self.resample_indices(w)
                    x_t.rots = x_t.rots[resampled_indices]
                    x_t.trans = x_t.trans[resampled_indices]

                    score = score[resampled_indices]
                    log_p_tilde_t = log_p_tilde_t[resampled_indices]
                    grad_log_p_tilde_t = grad_log_p_tilde_t[resampled_indices]

                    w[:] = 1 / K

                ## Recenter with respect to motif segment's center-of-mass
                ## (This is not necessary but makes for easier visual comparison when plotted)
                x_t.trans[:, :N_RESIDUES] -= torch.mean(
                    x_t.trans[:, OBSERVED_REGION], dim=1
                ).unsqueeze(1)

                t = torch.tensor([i] * K, device=self.device).long()

                x_t_plus_one = x_t
                t_plus_one = t

                # Propose next step
                ## Conditional score approximation s_{\theta}(x^{t + 1}, y=x_{M}^{0})
                cond_score = score + grad_log_p_tilde_t * twist_scale
                with self.model.with_score(cond_score):
                    x_t = self.model.reverse_diffuse(x_t_plus_one, t_plus_one, mask)

                x_trajectory.append(x_t)

                t = t_plus_one - 1
                log_p_tilde_t_plus_one = log_p_tilde_t

                # Update weights
                x_t.trans.requires_grad = True
                with torch.enable_grad():
                    score = self.model.score(x_t, t, mask)
                    with self.model.with_score(score):
                        x_zero_hat = self.model.predict_fully_denoised(x_t, t, mask)

                    log_p_tilde_t = log_likelihood(x_zero_hat, t)
                    (grad_log_p_tilde_t,) = torch.autograd.grad(
                        log_p_tilde_t.sum(), x_t.trans
                    )
                    log_p_tilde_t = log_p_tilde_t - torch.logsumexp(
                        log_p_tilde_t, dim=0
                    )
                x_t.trans = x_t.trans.detach()

                ## Reverse log likelihoods
                with self.model.with_score(score):
                    reverse_llik = self.model.reverse_log_likelihood(
                        x_t, x_t_plus_one, t_plus_one, mask, mask
                    )
                reverse_llik -= torch.logsumexp(reverse_llik, dim=0)

                with self.model.with_score(cond_score):
                    reverse_cond_llik = self.model.reverse_log_likelihood(
                        x_t, x_t_plus_one, t_plus_one, mask, mask
                    )
                reverse_cond_llik -= torch.logsumexp(reverse_cond_llik, dim=0)

                log_w = (reverse_llik + log_p_tilde_t) - (
                    reverse_cond_llik + log_p_tilde_t_plus_one
                )
                log_w = log_w - torch.max(log_w)
                log_sum_w = torch.logsumexp(log_w, dim=0)

                w *= torch.exp(log_w - log_sum_w)
                if torch.all(w == 0):
                    w[:] = 1 / K
                    ess = 0
                else:
                    w /= w.sum()
                    ess = (1 / torch.sum(w**2)).cpu()

                pf_stats["w"].append(w.cpu())
                pf_stats["ess"].append(ess)

        self.save_stats(pf_stats)

        return x_trajectory
