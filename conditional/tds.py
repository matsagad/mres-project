from conditional import register_conditional_method
from conditional.components.particle_filter import ParticleFilter, LikelihoodMethod
from conditional.wrapper import ConditionalWrapper, ConditionalWrapperConfig
from functools import partial
from model.diffusion import FrameDiffusionModel
import torch
from torch import Tensor
from tqdm import tqdm
from typing import Callable
from utils.resampling import get_resampling_method


class TDSConfig(ConditionalWrapperConfig):
    n_batches: int
    likelihood_method: LikelihoodMethod
    likelihood_sigma: float
    resampling_method: str
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
        likelihood_method: LikelihoodMethod = LikelihoodMethod.MASK,
        likelihood_sigma: float = 0.05,
        resampling_method: str = "residual",
        twist_scale: float = 1.0,
    ) -> "TDS":
        self.n_batches = n_batches
        self.likelihood_method = likelihood_method
        self.likelihood_sigma = likelihood_sigma
        self.resample_indices = get_resampling_method(resampling_method)
        self.twist_scale = twist_scale
        return self

    def sample_given_motif(
        self, mask: Tensor, motif: Tensor, motif_mask: Tensor
    ) -> Tensor:
        N_RESIDUES = (mask[0] == 1).sum().item()
        N_COORDS_PER_RESIDUE = 3
        D = N_RESIDUES * N_COORDS_PER_RESIDUE

        log_likelihood = self.get_log_likelihood(self.likelihood_method)
        if self.likelihood_method == LikelihoodMethod.MATRIX:
            A = []
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
            log_likelihood = partial(log_likelihood, A=A)

        return self.sample_conditional(
            mask, motif, motif_mask, log_likelihood, recenter_x=True
        )

    def sample_given_symmetry(self, mask: Tensor, symmetry: str) -> Tensor:
        A, y, y_mask = self._get_symmetric_constraints(mask, symmetry)

        assert self.likelihood_method == LikelihoodMethod.MATRIX, (
            f"Likelihood method '{self.likelihood_method}' is not supported for sampling symmetry."
            f" Make sure to set experiment.conditional_method.likelihood_method='{LikelihoodMethod.MATRIX}'"
        )
        log_likelihood = partial(self.get_log_likelihood(self.likelihood_method), A=[A])

        return self.sample_conditional(mask, y, y_mask, log_likelihood, recenter_x=True)

    def sample_conditional(
        self,
        mask: Tensor,
        y: Tensor,
        y_mask: Tensor,
        log_likelihood: Callable,
        recenter_x: bool = True,
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
        OBSERVED_REGION = torch.sum(y_mask, dim=0) == 1
        twist_scale = self.twist_scale

        # Set-up motif
        y_zero = self.model.coords_to_frames(y, y_mask)

        # Sample frames
        T = torch.tensor([N_TIMESTEPS - 1] * K, device=self.device).long()
        x_T = self.model.sample_frames(mask)
        x_T.trans.requires_grad = True
        alpha_bar_T = 1 - self.model.forward_variance[T[0]]

        # Initialise weights
        score = self.model.score(x_T, T, mask)
        ## Custom context manager for caching score to avoid repeated calls to model
        with self.model.with_score(score):
            x_zero_hat = self.model.predict_fully_denoised(x_T, T, mask)

        ## Calculate log likelihood and its gradient
        variance = 1 - alpha_bar_T + self.likelihood_sigma**2
        log_p_tilde_T = log_likelihood(x_zero_hat, y_zero, y_mask, variance)
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
                if recenter_x:
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
                alpha_bar_t = 1 - self.model.forward_variance[t[0]]
                x_t.trans.requires_grad = True
                with torch.enable_grad():
                    score = self.model.score(x_t, t, mask)
                    with self.model.with_score(score):
                        x_zero_hat = self.model.predict_fully_denoised(x_t, t, mask)

                    # We use Var(x_t | x_0) := (1 - alpha_bar_t) here but add
                    # some tiny sigma^2 since it goes to 0 as t goes to 0
                    variance_t = 1 - alpha_bar_t + self.likelihood_sigma**2
                    log_p_tilde_t = log_likelihood(
                        x_zero_hat, y_zero, y_mask, variance_t
                    )
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
