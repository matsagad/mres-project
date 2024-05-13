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
        noisy_motif: bool = False,
        particle_filter: bool = True,
        resample_indices: Callable = residual_resample,
        sigma: float = 0.05,
    ) -> "FPSSMC":
        self.noisy_motif = noisy_motif
        self.particle_filter = particle_filter
        self.resample_indices = resample_indices
        self.sigma = sigma
        return self

    def _3d_rot_matrices(self, theta: float) -> Tensor:
        rot_x = torch.tensor(
            [
                [1, 0, 0],
                [0, torch.cos(theta), -torch.sin(theta)],
                [0, torch.sin(theta), torch.cos(theta)],
            ]
        ).to(self.device)
        rot_y = torch.tensor(
            [
                [torch.cos(theta), 0, torch.sin(theta)],
                [0, 1, 0],
                [-torch.sin(theta), 0, torch.cos(theta)],
            ]
        ).to(self.device)
        rot_z = torch.tensor(
            [
                [torch.cos(theta), -torch.sin(theta), 0],
                [torch.sin(theta), torch.cos(theta), 0],
                [0, 0, 1],
            ]
        ).to(self.device)
        return rot_x, rot_y, rot_z

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
            mask, motif, motif_mask, A, recenter_protein=True
        )

    def sample_given_symmetry(self, mask: Tensor, symmetry: str) -> Tensor:
        N_RESIDUES = (mask[0] == 1).sum().item()
        N_COORDS_PER_RESIDUE = 3

        D = N_RESIDUES * N_COORDS_PER_RESIDUE
        SYM_GROUP_DELIM = "-"

        if symmetry[0] == "S":
            # Cyclic symmetry S-n
            N_SYMMETRIES = int(symmetry.split(SYM_GROUP_DELIM)[-1])
            N_FIXED_RESIDUES = (N_RESIDUES // N_SYMMETRIES) * N_SYMMETRIES

            d = N_FIXED_RESIDUES * N_COORDS_PER_RESIDUE

            theta = torch.tensor(2 * torch.pi / N_SYMMETRIES)
            rot_x, *_ = self._3d_rot_matrices(theta)

            A = torch.zeros((d, D), device=self.device)
            NCPR = N_COORDS_PER_RESIDUE
            for i in range(N_FIXED_RESIDUES):
                A[NCPR * i : NCPR * (i + 1), NCPR * i : NCPR * (i + 1)] = rot_x
            A[
                range(d),
                torch.arange(d).roll(-d // N_SYMMETRIES),
            ] -= 1
            y_mask = torch.zeros((1, N_RESIDUES), device=self.device)
            y_mask[:, :N_FIXED_RESIDUES] = 1
            y = torch.zeros((1, N_RESIDUES, N_COORDS_PER_RESIDUE), device=self.device)
            return self.sample_conditional(mask, y, y_mask, A)

        raise NotImplementedError()

    def sample_conditional(
        self,
        mask: Tensor,
        y: Frames,
        y_mask: Tensor,
        A: Tensor,
        recenter_protein: bool = False,
    ) -> Tensor:
        """
        Filtering Posterior Sampling with Sequential Monte Carlo as
        defined in the FPS paper: https://openreview.net/pdf?id=tplXNcHZs1
        (Dou & Song, 2024) but adapted for motif-scaffolding.
        """
        N_TIMESTEPS = self.model.n_timesteps
        N_COORDS_PER_RESIDUE = 3
        K, MAX_N_RESIDUES = mask.shape
        sigma = self.sigma

        N_RESIDUES = (mask[0] == 1).sum().item()
        N_OBSERVED = (y_mask[0] == 1).sum().item()
        D = N_RESIDUES * N_COORDS_PER_RESIDUE
        d = N_OBSERVED * N_COORDS_PER_RESIDUE
        assert d == A.shape[0] and D == A.shape[1]

        # (1) Generate sequence \{y^{t}\}^{T}_{t=0}
        y_frame = self.model.coords_to_frames(
            y.view(1, -1, N_COORDS_PER_RESIDUE), y_mask
        )
        y_sequence = [y_frame]

        forward_diffuse = (
            self.model.forward_diffuse
            if self.noisy_motif
            else self.model.forward_diffuse_deterministic
        )

        w = torch.ones(K, device=self.device) / K
        pf_stats = {"ess": [], "w": []}

        for i in tqdm(
            reversed(range(N_TIMESTEPS)),
            desc="Generating {y_t}",
            total=N_TIMESTEPS,
            disable=not self.verbose,
        ):
            t = torch.tensor([i] * y_frame.shape[0], device=self.device).long()
            y_frame = forward_diffuse(y_frame, t, y_mask)
            y_sequence.append(y_frame)

        # (2) Generate backward sequence \{x^{t}\}^{T}_{t=0}
        x_T = self.model.sample_frames(mask)
        x_sequence = [x_T]
        x_t = x_T

        for i in tqdm(
            reversed(range(N_TIMESTEPS - 1)),
            desc="Generating {x_t}",
            total=N_TIMESTEPS,
            disable=not self.verbose,
        ):
            t = torch.tensor([i + 1] * K, device=self.device).long()

            covariance_inverse = torch.zeros((D, D), device=self.device)
            covariance_inverse[range(D), range(D)] = 1 / (
                self.model.variance[t[:1]]
                * self.model.forward_variance[t[:1] - 1]
                / self.model.forward_variance[t[:1]]
            )
            mean = self.model.reverse_diffuse_deterministic(x_t, t, mask)
            if recenter_protein:
                # Translate mean so that motif segment is centred at zero
                mean.trans[:, :N_RESIDUES] -= torch.mean(
                    mean.trans[:, y_mask[0] == 1], dim=1
                ).unsqueeze(1)

            covariance_fps_inverse = covariance_inverse + (A.T @ A) / (
                sigma**2 * self.model.forward_variance[t[:1] - 1]
            )
            covariance_fps = torch.inverse(covariance_fps_inverse)

            mean_fps = (
                mean.trans[:, :N_RESIDUES].view(-1, D) @ covariance_inverse.T
                + (A.T @ y_sequence[t[:1] - 1].trans[:, y_mask[0] == 1].view(d))
                / (sigma**2 * self.model.forward_variance[t[:1] - 1])
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
