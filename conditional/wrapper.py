from abc import ABC, abstractmethod
from model.diffusion import FrameDiffusionModel
import os
from torch import Tensor
import torch
import tqdm
from typing import Dict, List, Tuple
from utils.path import out_dir
from utils.registry import ConfigOutline


class ConditionalWrapperConfig(ConfigOutline):
    pass


class ConditionalWrapper(ABC):
    def __init__(self, model: FrameDiffusionModel) -> None:
        self.model = model
        self.device = model.device
        self.verbose = True

    @property
    def device(self) -> str:
        return self._device

    @device.setter
    def device(self, _device: int) -> None:
        self._device = _device

    @property
    def verbose(self) -> bool:
        return self._verbose

    @verbose.setter
    def verbose(self, _verbose: bool) -> None:
        self._verbose = _verbose

    @property
    def supports_condition_on_motif(self) -> bool:
        return self._supports_condition_on_motif

    @supports_condition_on_motif.setter
    def supports_condition_on_motif(self, is_supported: bool) -> None:
        self._supports_condition_on_motif = is_supported

    @property
    def supports_condition_on_symmetry(self) -> bool:
        return self._supports_condition_on_symmetry

    @supports_condition_on_symmetry.setter
    def supports_condition_on_symmetry(self, is_supported: bool) -> None:
        self._supports_condition_on_symmetry = is_supported

    @abstractmethod
    def sample_given_motif(
        self, mask: Tensor, motif: Tensor, motif_mask: Tensor
    ) -> Tensor:
        """Sample conditioned on motif being present"""
        raise NotImplementedError

    @abstractmethod
    def sample_given_symmetry(self, mask: Tensor, symmetry: str) -> Tensor:
        """Sample conditioned on point symmetry"""
        raise NotImplementedError

    @abstractmethod
    def sample_given_motif_and_symmetry(
        self,
        mask: Tensor,
        motif: Tensor,
        motif_mask: Tensor,
        symmetry: str,
        fix_position: bool = False,
    ) -> Tensor:
        """Sample conditioned on point symmetry"""
        raise NotImplementedError

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

    def get_random_3d_rot_matrix(self, n: int) -> Tensor:
        z_axis = torch.eye(3)[-1]
        u1, u2, u3 = torch.rand((3, n))

        R_z = self._general_3d_rot_matrix(2 * torch.pi * u1, z_axis)
        v = torch.stack(
            [
                torch.cos(2 * torch.pi * u2) * torch.sqrt(u3),
                torch.sin(2 * torch.pi * u2) * torch.sqrt(u3),
                torch.sqrt(1 - u3),
            ]
        ).T
        H = torch.eye(3).unsqueeze(0) - 2 * v.unsqueeze(2) @ v.unsqueeze(1)
        return -H @ R_z

    def _get_symmetric_constraints(
        self, mask: Tensor, symmetry: str
    ) -> Tuple[Tensor, Tensor, Tensor]:
        MAX_N_RESIDUES = mask.shape[1]
        N_RESIDUES = (mask[0] == 1).sum().item()
        N_COORDS_PER_RESIDUE = 3
        x_axis, y_axis, z_axis = torch.eye(3)

        d = None
        D = N_RESIDUES * N_COORDS_PER_RESIDUE
        SYM_GROUP_DELIM = "-"
        symmetry_group = symmetry.split(SYM_GROUP_DELIM)[0]

        if symmetry_group == "C":
            # Cyclic symmetry C-n
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

        y_mask = torch.zeros((1, MAX_N_RESIDUES), device=self.device)
        y_mask[:, :N_FIXED_RESIDUES] = 1
        y = torch.zeros((1, MAX_N_RESIDUES, N_COORDS_PER_RESIDUE), device=self.device)

        return A, y, y_mask

    def _get_motif_and_symmetry_constraints(
        self,
        mask: Tensor,
        motif: Tensor,
        motif_mask: Tensor,
        symmetry: str,
        fix_position: bool = False,
    ) -> Tuple[List[Tensor], Tensor, Tensor]:
        N_MOTIF_RESIDUES = (motif_mask[0] == 1).sum()
        A_sym, y_sym, y_sym_mask = self._get_symmetric_constraints(mask, symmetry)
        N_COORDS_PER_RESIDUE = 3
        N_RESIDUES = (mask[0] == 1).sum()

        if fix_position:
            diag_motif = torch.diag(
                motif_mask[0, :N_RESIDUES].repeat_interleave(N_COORDS_PER_RESIDUE)
            )
            diag_motif = diag_motif[diag_motif.sum(1) > 0]
            A_sym[: N_COORDS_PER_RESIDUE * N_MOTIF_RESIDUES] += diag_motif
            y_sym[:, :N_MOTIF_RESIDUES] = motif[:, motif_mask[0] == 1]
            A = [A_sym]
            y = y_sym
            y_mask = y_sym_mask
        else:
            d = N_MOTIF_RESIDUES * N_COORDS_PER_RESIDUE
            D = N_RESIDUES * N_COORDS_PER_RESIDUE

            motif_indices_flat = torch.where(
                torch.repeat_interleave(
                    motif_mask[0, :N_RESIDUES] == 1, N_COORDS_PER_RESIDUE
                )
            )[0]
            A_motif = torch.zeros((d, D), device=self.device)
            A_motif[range(d), motif_indices_flat] = 1
            A = [A_sym, A_motif]
            y = torch.cat([y_sym, motif])
            y_mask = torch.cat([y_sym_mask, motif_mask])

        return A, y, y_mask

    def sample(self, mask: Tensor) -> Tensor:
        """Sample unconditionally"""

        if not self.model.setup:
            self.setup_schedule()

        x_T = self.model.sample_frames(mask)
        x_trajectory = [x_T]
        x_t = x_T

        with torch.no_grad():
            for i in tqdm.tqdm(
                reversed(range(self.model.n_timesteps)),
                desc="Reverse diffusing samples",
                total=self.model.n_timesteps,
                disable=not self.verbose,
            ):
                t = torch.tensor([i] * mask.shape[0], device=self.device).long()
                x_t = self.model.reverse_diffuse(x_t, t, mask)
                x_t.trans[:, mask[0] == 1] -= torch.mean(
                    x_t.trans[:, mask[0] == 1], dim=1
                ).unsqueeze(1)
                x_trajectory.append(x_t)

        return x_trajectory

    def save_stats(self, stats: Dict[str, any]) -> None:
        out = out_dir()
        os.makedirs(os.path.join(out, "stats"), exist_ok=True)
        for stat, values in stats.items():
            if isinstance(values, list) and not values:
                continue
            if isinstance(values, Tensor):
                tensor_values = values
            else:
                tensor_values = (
                    torch.stack(values)
                    if type(values[0]) == torch.Tensor
                    else torch.tensor(values)
                )
            torch.save(tensor_values, os.path.join(out, "stats", f"{stat}.pt"))
