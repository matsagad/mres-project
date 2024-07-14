from conditional import register_conditional_method
from conditional.particle_filter import ParticleFilter
from conditional.wrapper import ConditionalWrapper, ConditionalWrapperConfig
from enum import Enum
from model.diffusion import FrameDiffusionModel
from protein.frames import Frames
import torch
from torch import Tensor
from tqdm import tqdm
from typing import Callable
from utils.resampling import get_resampling_method


class TDSLikelihoodMethod(str, Enum):
    MASK = "mask"
    DISTANCE = "distance"


class TDSConfig(ConditionalWrapperConfig):
    n_batches: int
    likelihood_method: TDSLikelihoodMethod
    resampling_method: str
    sigma: float
    twist_scale: float


@register_conditional_method("tds", TDSConfig)
class TDS(ConditionalWrapper, ParticleFilter):

    def __init__(self, model: FrameDiffusionModel) -> None:
        super().__init__(model)
        self.with_config()

        self.supports_condition_on_motif = True
        self.supports_condition_on_symmetry = False

        # TDS does not benefit from having duplicate particles, so
        # can turn this optimisation off to avoid checking uniqueness.
        self.model.compute_unique_only = False

    def with_config(
        self,
        n_batches: int = 1,
        likelihood_method: TDSLikelihoodMethod = TDSLikelihoodMethod.MASK,
        resampling_method: str = "residual",
        sigma: float = 0.05,
        twist_scale: float = 1.0,
    ) -> "TDS":
        self.n_batches = n_batches
        self.likelihood_method = likelihood_method
        self.resample_indices = get_resampling_method(resampling_method)
        self.sigma = sigma
        self.twist_scale = twist_scale
        return self

    def get_likelihood(self, x_motif: Frames, motif_mask: Tensor) -> Callable:
        OBSERVED_REGION = motif_mask[0] == 1

        if self.likelihood_method == TDSLikelihoodMethod.MASK:

            def log_likelihood(x_zero_hat: Frames, t: Tensor) -> Tensor:
                centred_x_zero_hat_trans = x_zero_hat.trans[
                    :, OBSERVED_REGION
                ] - torch.mean(x_zero_hat.trans[:, OBSERVED_REGION], dim=1).unsqueeze(1)
                return -0.5 * (
                    (
                        (centred_x_zero_hat_trans - x_motif.trans[:, OBSERVED_REGION])
                        ** 2
                    )
                    / (self.model.forward_variance[t].view(-1, 1, 1) + self.sigma**2)
                ).sum(dim=(1, 2))

            return log_likelihood

        if self.likelihood_method == TDSLikelihoodMethod.DISTANCE:
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
                    / (self.model.forward_variance[t].view(-1, 1) + self.sigma**2)
                ).sum(axis=1)

            return log_likelihood

        raise KeyError(
            f"No such supported likelihood method: {self.likelihood_method}."
        )

    def sample_given_motif(
        self, mask: Tensor, motif: Tensor, motif_mask: Tensor
    ) -> Tensor:
        """
        Twisted Diffusion Sampler (for fixed motif inpainting) as defined
        in the paper: https://arxiv.org/pdf/2306.17775 (Wu et al., 2023)
        """
        N_BATCHES = self.n_batches
        N_TIMESTEPS = self.model.n_timesteps
        K, N_RESIDUES = mask.shape
        K_batch = K // N_BATCHES
        assert (
            K % N_BATCHES == 0
        ), f"Number of batches {N_BATCHES} does not divide number of particles {K}"
        OBSERVED_REGION = motif_mask[0] == 1
        twist_scale = self.twist_scale

        # Set-up motif
        x_motif = self.model.coords_to_frames(motif, motif_mask)
        log_likelihood = self.get_likelihood(x_motif, motif_mask)

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
        log_p_tilde_T_sum = torch.logsumexp(
            log_p_tilde_T.view(N_BATCHES, K_batch), dim=1
        ).repeat_interleave(K_batch)
        log_p_tilde_T = log_p_tilde_T - log_p_tilde_T_sum

        x_T.trans = x_T.trans.detach()

        with torch.no_grad():
            w = torch.exp(log_p_tilde_T).view(N_BATCHES, K_batch)
            w /= torch.sum(w, dim=1, keepdim=True)
            ess = 1 / torch.sum(w**2, dim=1).to(self.device)

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
                # Resample objects
                self.resample(
                    w,
                    ess,
                    [x_t.rots, x_t.trans, score, log_p_tilde_t, grad_log_p_tilde_t],
                )

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
                    log_p_tilde_t_sum = torch.logsumexp(
                        log_p_tilde_t.view(N_BATCHES, K_batch), dim=1
                    ).repeat_interleave(K_batch)
                    log_p_tilde_t = log_p_tilde_t - log_p_tilde_t_sum
                x_t.trans = x_t.trans.detach()

                ## Reverse log likelihoods
                with self.model.with_score(score):
                    reverse_llik = self.model.reverse_log_likelihood(
                        x_t, x_t_plus_one, t_plus_one, mask, mask
                    )

                with self.model.with_score(cond_score):
                    reverse_cond_llik = self.model.reverse_log_likelihood(
                        x_t, x_t_plus_one, t_plus_one, mask, mask
                    )

                log_w = (
                    (reverse_llik + log_p_tilde_t)
                    - (reverse_cond_llik + log_p_tilde_t_plus_one)
                ).view(N_BATCHES, K_batch)
                log_sum_w = torch.logsumexp(log_w, dim=1, keepdim=True)

                w *= torch.exp(log_w - log_sum_w)
                all_zeros = torch.all(w == 0, dim=1)

                w[all_zeros] = 1 / K_batch
                ess[all_zeros] = 0
                w[~all_zeros] /= w[~all_zeros].sum(dim=1, keepdim=True)
                ess[~all_zeros] = 1 / torch.sum(w[~all_zeros] ** 2, dim=1)

                pf_stats["w"].append(w.cpu())
                pf_stats["ess"].append(ess.cpu())

        self.save_stats(pf_stats)

        return x_trajectory

    def sample_given_symmetry(self, mask: Tensor, symmetry: str) -> Tensor:
        raise NotImplementedError("")
