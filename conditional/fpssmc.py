from conditional import register_conditional_method
from conditional.components.observation_generator import (
    LinearObservationGenerator,
    ObservationGenerationMethod,
)
from conditional.components.particle_filter import ParticleFilter, LikelihoodMethod
from conditional.wrapper import ConditionalWrapper, ConditionalWrapperConfig
from model.diffusion import FrameDiffusionModel
from protein.frames import Frames
import torch
from torch import Tensor
from tqdm import tqdm
from utils.resampling import get_resampling_method


class FPSSMCConfig(ConditionalWrapperConfig):
    n_batches: int
    fixed_motif: bool
    likelihood_sigma: float
    observed_sequence_method: ObservationGenerationMethod
    observed_sequence_noised: bool
    particle_filter: bool
    resampling_method: str


@register_conditional_method("fpssmc", FPSSMCConfig)
class FPSSMC(ConditionalWrapper, ParticleFilter, LinearObservationGenerator):

    def __init__(self, model: FrameDiffusionModel) -> None:
        super().__init__(model)
        self.with_config()

        self.supports_condition_on_motif = True
        self.supports_condition_on_symmetry = True

        self.model.compute_unique_only = True

    def with_config(
        self,
        n_batches: int = 1,
        fixed_motif: bool = True,
        likelihood_sigma: float = 0.1,
        observed_sequence_method: ObservationGenerationMethod = ObservationGenerationMethod.BACKWARD,
        observed_sequence_noised: bool = True,
        particle_filter: bool = True,
        resampling_method: str = "residual",
    ) -> "FPSSMC":
        self.n_batches = n_batches
        self.fixed_motif = fixed_motif
        self.likelihood_sigma = likelihood_sigma
        self.observed_sequence_method = observed_sequence_method
        self.observed_sequence_noised = observed_sequence_noised
        self.particle_filter = particle_filter
        self.resample_indices = get_resampling_method(resampling_method)
        return self

    def _general_3d_rot_matrix(self, thetas: Tensor, axis: Tensor) -> Tensor:
        assert len(axis) == 3
        u = axis / (axis**2).sum()
        u_x, u_y, u_z = u

        _cos = torch.cos(thetas)
        _sin = torch.sin(thetas)

        R = torch.empty((len(thetas), 3, 3))
        R[:, 0] = torch.stack(
            [
                _cos + (u_x**2) * (1 - _cos),
                u_x * u_y * (1 - _cos) - u_z * _sin,
                u_x * u_z * (1 - _cos) + u_y * _sin,
            ]
        ).T
        R[:, 1] = torch.stack(
            [
                u_y * u_x * (1 - _cos) + u_z * _sin,
                _cos + (u_y**2) * (1 - _cos),
                u_y * u_z * (1 - _cos) - u_x * _sin,
            ]
        ).T
        R[:, 2] = torch.stack(
            [
                u_z * u_x * (1 - _cos) - u_y * _sin,
                u_z * u_y * (1 - _cos) + u_x * _sin,
                _cos + (u_z**2) * (1 - _cos),
            ]
        ).T

        return R

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
            mask, motif, motif_mask, A, recenter_x=True
        )

    def sample_given_symmetry(self, mask: Tensor, symmetry: str) -> Tensor:
        N_RESIDUES = (mask[0] == 1).sum().item()
        N_COORDS_PER_RESIDUE = 3
        x_axis, y_axis, z_axis = torch.eye(3)

        d = None
        D = N_RESIDUES * N_COORDS_PER_RESIDUE
        SYM_GROUP_DELIM = "-"
        symmetry_group = symmetry.split(SYM_GROUP_DELIM)[0]

        if symmetry_group == "S":
            # Cyclic symmetry S-n
            N_SYMMETRIES = int(symmetry.split(SYM_GROUP_DELIM)[-1])
            N_RESIDUES_PER_DOMAIN = N_RESIDUES // N_SYMMETRIES
            N_FIXED_RESIDUES = N_RESIDUES_PER_DOMAIN * N_SYMMETRIES

            d = N_FIXED_RESIDUES * N_COORDS_PER_RESIDUE

            thetas = 2 * torch.pi * torch.arange(N_SYMMETRIES).float() / N_SYMMETRIES
            R_z = self._general_3d_rot_matrix(thetas, z_axis)
            F = R_z

        elif symmetry_group == "D":
            # Dihedral symmetry D-n
            N_SYMMETRIES = 2 * int(symmetry.split(SYM_GROUP_DELIM)[-1])
            N_RESIDUES_PER_DOMAIN = N_RESIDUES // N_SYMMETRIES
            N_FIXED_RESIDUES = N_RESIDUES_PER_DOMAIN * N_SYMMETRIES

            d = N_FIXED_RESIDUES * N_COORDS_PER_RESIDUE

            thetas = (
                2
                * torch.pi
                * torch.arange(N_SYMMETRIES // 2).float()
                / (N_SYMMETRIES // 2)
            )
            R_x_pi = self._general_3d_rot_matrix(torch.tensor([torch.pi]), x_axis)
            R_z = self._general_3d_rot_matrix(thetas, z_axis)
            S = torch.einsum("sij,tjk->tik", R_x_pi, R_z)
            F = torch.concatenate((R_z, S))

        assert d is not None, f"Unsupported symmetry group chosen: {symmetry_group}"

        F = F.to(self.device)
        A = torch.zeros((d, D), device=self.device)
        NCPR = N_COORDS_PER_RESIDUE
        for k, F_k in enumerate(F):
            offset = NCPR * N_RESIDUES_PER_DOMAIN * k
            for i in range(N_RESIDUES_PER_DOMAIN):
                A[
                    offset + NCPR * i : offset + NCPR * (i + 1),
                    NCPR * i : NCPR * (i + 1),
                ] = F_k

        assert d <= D
        A[range(d), range(d)] -= 1

        y_mask = torch.zeros((1, N_RESIDUES), device=self.device)
        y_mask[:, :N_FIXED_RESIDUES] = 1
        y = torch.zeros((1, N_RESIDUES, N_COORDS_PER_RESIDUE), device=self.device)

        return self.sample_conditional(
            mask, y, y_mask, A, recenter_y=False, recenter_x=True
        )

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
        Filtering Posterior Sampling with Sequential Monte Carlo as
        defined in the FPS paper: https://openreview.net/pdf?id=tplXNcHZs1
        (Dou & Song, 2024)

        TODO: Generalise for d != N x 3, i.e. a non-3D projection.
        """
        N_BATCHES = self.n_batches
        N_TIMESTEPS = self.model.n_timesteps
        N_COORDS_PER_RESIDUE = 3
        K, MAX_N_RESIDUES = mask.shape
        K_batch = K // N_BATCHES
        assert (
            K % N_BATCHES == 0
        ), f"Number of batches {N_BATCHES} does not divide number of particles {K}"
        sigma = self.likelihood_sigma

        OBSERVED_REGION = y_mask[0] == 1
        N_RESIDUES = (mask[0] == 1).sum().item()
        N_OBSERVED = (OBSERVED_REGION).sum().item()
        D = N_RESIDUES * N_COORDS_PER_RESIDUE
        d = N_OBSERVED * N_COORDS_PER_RESIDUE
        assert d == A.shape[0] and D == A.shape[1]

        # (1) Generate sequence {y_t}
        if self.observed_sequence_method == ObservationGenerationMethod.BACKWARD:
            epsilon_T = self.model.sample_frames(mask[:1])
            x_T = self.model.coords_to_frames(
                torch.tile(epsilon_T.trans, (K, 1, 1)), mask
            )
        else:
            x_T = self.model.sample_frames(mask)

        y_zero = self.model.coords_to_frames(
            y.view(1, -1, N_COORDS_PER_RESIDUE), y_mask
        )
        y_sequence = self.generate_observed_sequence(
            mask, y_zero, y_mask, A, x_T, recenter_y=recenter_y
        )

        w = torch.ones((N_BATCHES, K_batch), device=self.device) / K_batch
        ess = torch.zeros(N_BATCHES, device=self.device)
        pf_stats = {"ess": [], "w": []}

        # (2) Generate sequence {x_t}
        log_likelihood = self.get_log_likelihood(LikelihoodMethod.MATRIX)

        x_sequence = [x_T]
        x_t = x_T
        A_T_A = A.T @ A

        for i in tqdm(
            reversed(range(N_TIMESTEPS)),
            desc="Generating {x_t}",
            total=N_TIMESTEPS,
            disable=not self.verbose,
        ):
            t = torch.tensor([i] * K, device=self.device).long()

            alpha_bar_t = 1 - self.model.forward_variance[t[:1]]

            # Propose next step
            covariance_inverse = torch.zeros((D, D), device=self.device)
            covariance_inverse[range(D), range(D)] = 1 / self.model.variance[t[:1]]

            with torch.no_grad():
                score = self.model.score(x_t, t, mask)
            with self.model.with_score(score):
                mean = self.model.reverse_diffuse_deterministic(x_t, t, mask)
            if recenter_x:
                # Translate mean so that motif segment is centred at zero
                mean.trans[:, :N_RESIDUES] -= torch.mean(
                    mean.trans[:, OBSERVED_REGION], dim=1
                ).unsqueeze(1)

            covariance_fps_inverse = covariance_inverse + A_T_A / (
                sigma**2 * alpha_bar_t
            )
            covariance_fps = torch.inverse(covariance_fps_inverse)

            mean_fps = (
                mean.trans[:, :N_RESIDUES].view(-1, D) @ covariance_inverse.T
                + (A.T @ y_sequence[t[0]].trans[:, OBSERVED_REGION].view(d))
                / (sigma**2 * alpha_bar_t)
            ) @ covariance_fps.T

            mvn = torch.distributions.MultivariateNormal(
                mean_fps, (self.model.noise_scale**2) * covariance_fps
            )

            x_bar_t_trans = torch.empty(
                (K, MAX_N_RESIDUES, N_COORDS_PER_RESIDUE), device=self.device
            )
            x_bar_t_trans[:, N_RESIDUES:] = 0
            x_bar_t_trans[:, :N_RESIDUES] = mvn.sample((1,)).view(
                K, N_RESIDUES, N_COORDS_PER_RESIDUE
            )
            x_bar_t = self.model.coords_to_frames(x_bar_t_trans, mask)

            if not self.particle_filter:
                x_t = x_bar_t
                x_sequence.append(x_t)
                continue

            # Resample particles
            with self.model.with_score(score):
                reverse_llik = self.model.reverse_log_likelihood(
                    x_bar_t, x_t, t, mask, mask
                )
            reverse_cond_llik = mvn.log_prob(x_bar_t.trans[:, :N_RESIDUES].view(-1, D))

            if recenter_x:
                # Translate mean so that motif segment is centred at zero
                x_bar_t.trans[:, :N_RESIDUES] -= torch.mean(
                    x_bar_t.trans[:, OBSERVED_REGION], dim=1
                ).unsqueeze(1)
            variance_t = alpha_bar_t * self.likelihood_sigma**2
            y_llik = log_likelihood(x_bar_t, y_sequence[t[0]], y_mask, A, variance_t)

            log_w = (reverse_llik + y_llik - reverse_cond_llik).view(N_BATCHES, K_batch)
            log_sum_w = torch.logsumexp(log_w, dim=1, keepdim=True)

            w *= torch.exp(log_w - log_sum_w)
            all_zeros = torch.all(w == 0, dim=1)

            w[all_zeros] = 1 / K_batch
            ess[all_zeros] = 0
            w[~all_zeros] /= w[~all_zeros].sum(dim=1, keepdim=True)
            ess[~all_zeros] = 1 / torch.sum(w[~all_zeros] ** 2, dim=1)

            pf_stats["ess"].append(ess.cpu())
            pf_stats["w"].append(w.cpu())

            x_t = x_bar_t

            self.resample(w, ess, [x_t.rots, x_t.trans])

            x_sequence.append(x_t)

        if self.particle_filter:
            self.save_stats(pf_stats)

        return x_sequence
