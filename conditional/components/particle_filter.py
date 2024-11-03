from enum import Enum
from protein.frames import Frames, compute_frenet_frames
from protein.alignment import rmsd_kabsch
from torch import Tensor
import torch
from typing import Callable, List


class LikelihoodMethod(str, Enum):
    MASK = "mask"
    MATRIX = "matrix"
    DISTANCE = "distance"
    FRAME_BASED_DISTANCE = "frame_based_distance"
    RMSD = "rmsd"
    FAPE = "fape"


class LikelihoodReduction(Enum):
    PRODUCT = 1  # Product of experts
    SUM = 2  # Mixture of experts
    NONE = 3  # None (for debugging likelihood values)


class ParticleFilter:
    """
    Common subroutines for particle filtering algorithms.
    """

    @property
    def resample_indices(self) -> Callable[[Tensor], Tensor]:
        return self._resample_indices

    @resample_indices.setter
    def resample_indices(self, _resample_indices: Callable[[Tensor], Tensor]) -> None:
        self._resample_indices = _resample_indices

    def resample(self, w: Tensor, ess: Tensor, objects: List[Tensor]) -> None:
        n_batches, K_batch = w.shape
        K = n_batches * K_batch

        need_resampling = ess.cpu() <= 0.5 * K_batch
        w_to_resample = w[need_resampling]

        if w_to_resample.numel() != 0:
            n_to_resample = need_resampling.sum()
            resampled_indices = self.resample_indices(w_to_resample).cpu()

            # Find source and destination ranges.
            batch_range = K_batch * need_resampling.nonzero().view(
                -1, 1
            ) + torch.arange(K_batch).tile((n_to_resample, 1))
            resampled_range = batch_range[
                torch.arange(n_to_resample)
                .repeat_interleave(K_batch)
                .view(n_to_resample, K_batch),
                resampled_indices,
            ]

            # Here, we require each object to have leading dimension K and
            # have each batch be grouped together. E.g. [0,0,1,1,2,2,3,3].
            for obj in objects:
                assert obj.shape[0] == K
                obj[batch_range] = obj[resampled_range]

            w[need_resampling] = 1 / K_batch

    def get_log_likelihood(self, likelihood_method) -> Callable:

        if likelihood_method == LikelihoodMethod.MASK:

            def log_likelihood(
                x_t: Frames,
                y_t: Frames,
                y_mask: Tensor,
                variance: float,
                reduce: LikelihoodReduction = LikelihoodReduction.PRODUCT,
            ) -> Tensor:
                log_likelihoods = []
                for i in range(len(y_mask)):
                    OBSERVED_REGION = y_mask[i] == 1
                    com_offset = torch.mean(
                        x_t.trans[:, OBSERVED_REGION], dim=1, keepdim=True
                    )

                    llik = -0.5 * (
                        (
                            (
                                y_t.trans[i : i + 1, OBSERVED_REGION]
                                - (x_t.trans[:, OBSERVED_REGION] - com_offset)
                            )
                            ** 2
                        )
                        / variance
                    ).sum(dim=(1, 2))
                    log_likelihoods.append(llik)

                if reduce == LikelihoodReduction.PRODUCT:
                    return sum(log_likelihoods)
                if reduce == LikelihoodReduction.SUM:
                    return torch.logsumexp(torch.stack(log_likelihoods), dim=0)
                return torch.stack(log_likelihoods)

            return log_likelihood

        if likelihood_method == LikelihoodMethod.MATRIX:

            def log_likelihood(
                x_t: Frames,
                y_t: Frames,
                y_mask: Tensor,
                variance: float,
                A: List[Tensor],
                reduce: LikelihoodReduction = LikelihoodReduction.PRODUCT,
            ) -> Tensor:
                N_COORDS_PER_RESIDUE = 3
                log_likelihoods = []
                for i in range(len(y_mask)):
                    d, D = A[i].size()
                    N_RESIDUES = D // N_COORDS_PER_RESIDUE
                    OBSERVED_REGION = y_mask[i] == 1
                    com_offset = torch.mean(
                        x_t.trans[:, OBSERVED_REGION], dim=1
                    ).unsqueeze(1)

                    llik = -0.5 * (
                        (
                            (
                                y_t.trans[i : i + 1, OBSERVED_REGION].view(-1, d)
                                - (x_t.trans[:, :N_RESIDUES] - com_offset).view(-1, D)
                                @ A[i].T
                            )
                            ** 2
                        )
                        / variance
                    ).sum(dim=1)
                    log_likelihoods.append(llik)

                if reduce == LikelihoodReduction.PRODUCT:
                    return sum(log_likelihoods)
                if reduce == LikelihoodReduction.SUM:
                    return torch.logsumexp(torch.stack(log_likelihoods), dim=0)
                return torch.stack(log_likelihoods)

            return log_likelihood

        if likelihood_method == LikelihoodMethod.DISTANCE:

            def log_likelihood(
                x_t: Frames,
                y_t: Frames,
                y_mask: Tensor,
                variance: float,
                reduce: LikelihoodReduction = LikelihoodReduction.PRODUCT,
            ) -> Tensor:
                log_likelihoods = []
                dist_x_full = torch.cdist(x_t.trans, x_t.trans)
                for i in range(len(y_mask)):
                    y_t_trans = y_t.trans[i : i + 1]
                    OBSERVED_REGION = y_mask[i] == 1
                    N_OBSERVED = torch.sum(OBSERVED_REGION)
                    _i, _j = torch.triu_indices(N_OBSERVED, N_OBSERVED, offset=1)

                    dist_y = torch.cdist(
                        y_t_trans[:, OBSERVED_REGION], y_t_trans[:, OBSERVED_REGION]
                    )[:, _i, _j].view(1, (N_OBSERVED * (N_OBSERVED - 1)) // 2)
                    dist_x = dist_x_full[:, OBSERVED_REGION][:, :, OBSERVED_REGION][
                        :, _i, _j
                    ].view(-1, (N_OBSERVED * (N_OBSERVED - 1)) // 2)

                    llik = (-0.5 * ((dist_y - dist_x) ** 2) / variance).sum(dim=1)
                    log_likelihoods.append(llik)

                if reduce == LikelihoodReduction.PRODUCT:
                    return sum(log_likelihoods)
                if reduce == LikelihoodReduction.SUM:
                    return torch.logsumexp(torch.stack(log_likelihoods), dim=0)
                return torch.stack(log_likelihoods)

            return log_likelihood

        if likelihood_method == LikelihoodMethod.FRAME_BASED_DISTANCE:

            def log_likelihood(
                x_t: Frames,
                y_t: Frames,
                y_mask: Tensor,
                variance: float,
                rot_likelihood_scale: float = 64.0,
                reduce: LikelihoodReduction = LikelihoodReduction.PRODUCT,
            ) -> Tensor:
                log_likelihoods = []
                K = x_t.trans.shape[0]
                x_t_trans_centred = x_t.trans - torch.mean(
                    x_t.trans, dim=1, keepdim=True
                )
                dist_x_full = torch.cdist(x_t_trans_centred, x_t_trans_centred)
                for i in range(len(y_mask)):
                    y_t_trans = y_t.trans[i : i + 1]
                    OBSERVED_REGION = y_mask[i] == 1
                    N_OBSERVED = torch.sum(OBSERVED_REGION)
                    _i, _j = torch.triu_indices(N_OBSERVED, N_OBSERVED, offset=1)

                    dist_y = torch.cdist(
                        y_t_trans[:, OBSERVED_REGION], y_t_trans[:, OBSERVED_REGION]
                    )[:, _i, _j].view(1, (N_OBSERVED * (N_OBSERVED - 1)) // 2)
                    dist_x = dist_x_full[:, OBSERVED_REGION][:, :, OBSERVED_REGION][
                        :, _i, _j
                    ].view(-1, (N_OBSERVED * (N_OBSERVED - 1)) // 2)

                    dist_llik = (-0.5 * ((dist_y - dist_x) ** 2) / variance).sum(dim=1)

                    x_view = x_t_trans_centred[:, OBSERVED_REGION]
                    y_view = y_t_trans[:, OBSERVED_REGION]

                    x_rot_mats = compute_frenet_frames(
                        x_view, torch.ones(x_view.shape[:2])
                    )
                    y_rot_mats = compute_frenet_frames(
                        y_view, torch.ones(y_view.shape[:2])
                    )
                    # (K, N_OBSERVED, N_OBSERVED, 3, 3)
                    R_diff_x = torch.einsum(
                        "ijklm,iknmo->ijnlo",
                        x_rot_mats.view(K, N_OBSERVED, 1, 3, 3).transpose(3, 4),
                        x_rot_mats.view(K, 1, N_OBSERVED, 3, 3),
                    )
                    # (1, N_OBSERVED, N_OBSERVED, 3, 3)
                    R_diff_y = torch.einsum(
                        "ijklm,iknmo->ijnlo",
                        y_rot_mats.view(1, N_OBSERVED, 1, 3, 3).transpose(3, 4),
                        y_rot_mats.view(1, 1, N_OBSERVED, 3, 3),
                    )
                    # (K, 1, N_OBSERVED, N_OBSERVED, 3, 3)
                    R_diff_x_y = torch.einsum(
                        "ijklmn,joklnp->ioklmp",
                        R_diff_x.view(K, 1, N_OBSERVED, N_OBSERVED, 3, 3),
                        R_diff_y.view(1, 1, N_OBSERVED, N_OBSERVED, 3, 3).transpose(
                            4, 5
                        ),
                    )
                    # (K, N_OBSERVED, N_OBSERVED)
                    cos_angle_diff = (
                        R_diff_x_y[:, 0, :, :, torch.arange(3), torch.arange(3)].sum(
                            dim=3
                        )
                        - 1
                    ) / 2

                    rot_llik = (-0.5 * ((cos_angle_diff - 1) ** 2) / variance).sum(
                        dim=(1, 2)
                    ) / 2
                    # Divide by two since rot_llik computes both x1 - x2 and x2 - x1

                    llik = dist_llik + rot_likelihood_scale * rot_llik
                    log_likelihoods.append(llik)

                if reduce == LikelihoodReduction.PRODUCT:
                    return sum(log_likelihoods)
                if reduce == LikelihoodReduction.SUM:
                    return torch.logsumexp(torch.stack(log_likelihoods), dim=0)
                return torch.stack(log_likelihoods)

            return log_likelihood

        if likelihood_method == LikelihoodMethod.RMSD:

            def log_likelihood(
                x_t: Frames,
                y_t: Frames,
                y_mask: Tensor,
                variance: float,
                reduce: LikelihoodReduction = LikelihoodReduction.PRODUCT,
            ) -> Tensor:
                log_likelihoods = []
                for i in range(len(y_mask)):
                    OBSERVED_REGION = y_mask[i] == 1
                    rmsd = rmsd_kabsch(
                        x_t.trans[:, OBSERVED_REGION],
                        y_t.trans[i : i + 1, OBSERVED_REGION],
                    )
                    log_likelihoods.append(
                        -0.5 * OBSERVED_REGION.sum() * (rmsd**2) / variance
                    )

                if reduce == LikelihoodReduction.PRODUCT:
                    return sum(log_likelihoods)
                if reduce == LikelihoodReduction.SUM:
                    return torch.logsumexp(torch.stack(log_likelihoods), dim=0)
                return torch.stack(log_likelihoods)

            return log_likelihood

        if likelihood_method == LikelihoodMethod.FAPE:

            def log_likelihood(
                x_t: Frames,
                y_t: Frames,
                y_mask: Tensor,
                variance: float,
                reduce: LikelihoodReduction = LikelihoodReduction.PRODUCT,
            ) -> Tensor:
                log_likelihoods = []
                for i in range(len(y_mask)):
                    OBSERVED_REGION = y_mask[i] == 1
                    y_t_trans = y_t.trans[i : i + 1, OBSERVED_REGION]
                    x_t_trans = x_t.trans[:, OBSERVED_REGION]

                    y_t_rots = compute_frenet_frames(
                        y_t_trans, torch.ones(y_t_trans.shape[:2])
                    )
                    x_t_rots = compute_frenet_frames(
                        x_t_trans, torch.ones(x_t_trans.shape[:2])
                    )

                    y_t_trans_diff = y_t_trans.unsqueeze(1) - y_t_trans.unsqueeze(2)
                    x_t_trans_diff = x_t_trans.unsqueeze(1) - x_t_trans.unsqueeze(2)

                    y_res = torch.einsum("ijlk,ijml->ijmk", y_t_rots, y_t_trans_diff)
                    x_res = torch.einsum("ijlk,ijml->ijmk", x_t_rots, x_t_trans_diff)

                    loss_fape = (
                        -0.5
                        * (
                            torch.sum(
                                (x_res - y_res) ** 2,
                                dim=(1, 2, 3),
                            )
                            / OBSERVED_REGION.sum()
                        )
                        / variance
                    )
                    log_likelihoods.append(loss_fape)

                if reduce == LikelihoodReduction.PRODUCT:
                    return sum(log_likelihoods)
                if reduce == LikelihoodReduction.SUM:
                    return torch.logsumexp(torch.stack(log_likelihoods), dim=0)
                return torch.stack(log_likelihoods)

            return log_likelihood

        raise KeyError(
            f"No such supported likelihood method: {self.likelihood_method}."
        )
