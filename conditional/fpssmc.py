from conditional.wrapper import ConditionalWrapper
from model.diffusion import FrameDiffusionModel
from protein.frames import Frames
import torch
from torch import Tensor
from tqdm import tqdm
from typing import Callable
from utils.resampling import residual_resample


class FPSSMC(ConditionalWrapper):

    def __init__(self, model: FrameDiffusionModel) -> None:
        super().__init__(model)
        self.with_config()

    def with_config(
        self,
        noisy_y: bool = False,
        particle_filter: bool = True,
        resample_indices: Callable = residual_resample,
        sigma: float = 0.05,
    ) -> "FPSSMC":
        self.noisy_y = noisy_y
        self.particle_filter = particle_filter
        self.resample_indices = resample_indices
        self.sigma = sigma
        return self

    def _3d_rot_matrices(self, thetas: Tensor) -> Tensor:
        assert len(thetas) == 1

        _cos = torch.cos(thetas)
        _sin = torch.sin(thetas)
        zero = torch.zeros(len(thetas))
        one = torch.ones(len(thetas))

        R_x = torch.empty((len(thetas), 3, 3))
        R_x[:, 0] = torch.stack([one, zero, zero]).T
        R_x[:, 1] = torch.stack([zero, _cos, -_sin]).T
        R_x[:, 2] = torch.stack([zero, _sin, _cos]).T

        R_y = torch.empty((len(thetas), 3, 3))
        R_y[:, 0] = torch.stack([_cos, zero, _sin]).T
        R_y[:, 1] = torch.stack([zero, one, zero]).T
        R_y[:, 2] = torch.stack([-_sin, zero, _cos]).T

        R_z = torch.empty((len(thetas), 3, 3))
        R_z[:, 0] = torch.stack([_cos, -_sin, zero]).T
        R_z[:, 1] = torch.stack([_sin, _cos, zero]).T
        R_z[:, 2] = torch.stack([zero, zero, one]).T

        return R_x, R_y, R_z

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
            _, _, R_z = self._3d_rot_matrices(thetas)
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
            _, R_y_pi, _ = self._3d_rot_matrices(torch.tensor([torch.pi]))
            _, _, R_z = self._3d_rot_matrices(thetas)
            S = torch.einsum("sij,tjk->tik", R_y_pi, R_z)
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
        return self.sample_conditional(mask, y, y_mask, A, recenter_x=True)

    def sample_conditional(
        self,
        mask: Tensor,
        y: Frames,
        y_mask: Tensor,
        A: Tensor,
        recenter_x: bool = True,
    ) -> Tensor:
        """
        Filtering Posterior Sampling with Sequential Monte Carlo as
        defined in the FPS paper: https://openreview.net/pdf?id=tplXNcHZs1
        (Dou & Song, 2024)
        """
        N_TIMESTEPS = self.model.n_timesteps
        N_COORDS_PER_RESIDUE = 3
        K, MAX_N_RESIDUES = mask.shape
        sigma = self.sigma

        OBSERVED_REGION = y_mask[0] == 1
        N_RESIDUES = (mask[0] == 1).sum().item()
        N_OBSERVED = (OBSERVED_REGION).sum().item()
        D = N_RESIDUES * N_COORDS_PER_RESIDUE
        d = N_OBSERVED * N_COORDS_PER_RESIDUE
        assert d == A.shape[0] and D == A.shape[1]

        # (1) Generate sequence \{y^{t}\}^{T}_{t=0}
        y_0 = self.model.coords_to_frames(y.view(1, -1, N_COORDS_PER_RESIDUE), y_mask)

        epsilon_T = self.model.sample_frames(mask[:1])
        y_T_trans = torch.zeros(y_0.trans.shape, device=self.device)
        y_T_trans[:, OBSERVED_REGION] = (
            epsilon_T.trans[:, :N_RESIDUES].view(D) @ A.T
        ).view(1, -1, N_COORDS_PER_RESIDUE)
        y_T = self.model.coords_to_frames(y_T_trans, y_mask)

        y_t = y_T
        y_sequence = [y_T]

        w = torch.ones(K, device=self.device) / K
        pf_stats = {"ess": [], "w": []}

        for i in tqdm(
            reversed(range(N_TIMESTEPS)),
            desc="Generating {y_t}",
            total=N_TIMESTEPS,
            disable=not self.verbose,
        ):
            t = torch.tensor([i] * y_t.shape[0], device=self.device).long()

            alpha_bar_t = 1 - self.model.forward_variance[t]
            alpha_bar_t_minus_one = 1 - self.model.forward_variance[t - 1]

            c = self.model.variance[t] / (1 - alpha_bar_t)
            p_t = torch.sqrt((1 - c) * (1 - alpha_bar_t_minus_one) / (1 - alpha_bar_t))
            q_t = torch.sqrt(c * (1 - alpha_bar_t_minus_one))

            y_t_minus_one_trans = torch.zeros(y_0.trans.shape, device=self.device)
            y_t_minus_one_trans[:, OBSERVED_REGION] = torch.sqrt(
                alpha_bar_t_minus_one
            ) * y_0.trans[:, OBSERVED_REGION] + p_t * (
                y_t.trans[:, OBSERVED_REGION]
                - torch.sqrt(alpha_bar_t) * y_0.trans[:, OBSERVED_REGION]
            )

            if self.noisy_y:
                y_t_minus_one_trans[:, OBSERVED_REGION] += q_t * (
                    A @ torch.randn((D,), device=self.device)
                ).view(-1, N_COORDS_PER_RESIDUE)
            y_t_minus_one_trans[:, OBSERVED_REGION] -= torch.mean(
                y_t_minus_one_trans[:, OBSERVED_REGION], dim=1, keepdim=True
            )
            y_t_minus_one = self.model.coords_to_frames(y_t_minus_one_trans, y_mask)

            y_t = y_t_minus_one
            y_sequence.append(y_t)

        y_sequence.append(y_0)
        y_sequence = y_sequence[::-1]

        # (2) Generate backward sequence \{x^{t}\}^{T}_{t=0}
        x_T = self.model.coords_to_frames(torch.tile(epsilon_T.trans, (K, 1, 1)), mask)
        x_sequence = [x_T]
        x_t = x_T

        for i in tqdm(
            reversed(range(N_TIMESTEPS)),
            desc="Generating {x_t}",
            total=N_TIMESTEPS,
            disable=not self.verbose,
        ):
            t = torch.tensor([i] * K, device=self.device).long()

            covariance_inverse = torch.zeros((D, D), device=self.device)
            covariance_inverse[range(D), range(D)] = 1 / self.model.variance[t[:1]]
            mean = self.model.reverse_diffuse_deterministic(x_t, t, mask)
            if recenter_x:
                # Translate mean so that motif segment is centred at zero
                mean.trans[:, :N_RESIDUES] -= torch.mean(
                    mean.trans[:, y_mask[0] == 1], dim=1
                ).unsqueeze(1)

            covariance_fps_inverse = covariance_inverse + (A.T @ A) / (
                sigma**2 * self.model.forward_variance[t[:1]]
            )
            covariance_fps = torch.inverse(covariance_fps_inverse)

            mean_fps = (
                mean.trans[:, :N_RESIDUES].view(-1, D) @ covariance_inverse.T
                + (A.T @ y_sequence[t[:1]].trans[:, y_mask[0] == 1].view(d))
                / (sigma**2 * self.model.forward_variance[t[:1]])
            ) @ covariance_fps.T

            mvn = torch.distributions.MultivariateNormal(mean_fps, covariance_fps)

            x_bar_t_trans = torch.empty(
                (K, MAX_N_RESIDUES, N_COORDS_PER_RESIDUE), device=self.device
            )
            x_bar_t_trans[:, N_RESIDUES:] = 0
            x_bar_t_trans[:, :N_RESIDUES] = mvn.sample((1,)).view(
                K, N_RESIDUES, N_COORDS_PER_RESIDUE
            )
            x_bar_t = self.model.coords_to_frames(x_bar_t_trans, mask)
            if recenter_x:
                # Translate mean so that motif segment is centred at zero
                x_bar_t.trans[:, :N_RESIDUES] -= torch.mean(
                    x_bar_t.trans[:, y_mask[0] == 1], dim=1
                ).unsqueeze(1)

            if not self.particle_filter:
                x_t = x_bar_t
                x_sequence.append(x_t)
                continue

            # Resample particles
            reverse_llik = self.model.reverse_log_likelihood(
                x_bar_t, x_t, t, mask, mask
            )
            y_llik = (
                -0.5
                * (
                    y_sequence[t[:1]].trans[:, y_mask[0] == 1].view(-1, d)
                    - x_bar_t.trans[:, :N_RESIDUES].view(-1, D) @ A.T
                )
                ** 2
                / (sigma**2 * self.model.forward_variance[t[:1]])
            ).sum(axis=1)
            reverse_cond_llik = mvn.log_prob(x_bar_t.trans[:, :N_RESIDUES].view(-1, D))

            log_w = reverse_llik + y_llik - reverse_cond_llik
            log_sum_w = torch.logsumexp(log_w, dim=0)
            w = torch.exp(log_w - log_sum_w)
            w /= w.sum()

            ess = (1 / torch.sum(w**2)).cpu()

            pf_stats["ess"].append(ess)
            pf_stats["w"].append(w.cpu())

            x_t = x_bar_t

            if ess < 0.5 * K:
                resampled_indices = self.resample_indices(w)
                x_t.rots = x_t.rots[resampled_indices]
                x_t.trans = x_t.trans[resampled_indices]
                w = torch.ones(K, device=self.device) / K

            x_sequence.append(x_t)

        if self.particle_filter:
            self.save_stats(pf_stats)

        return x_sequence
