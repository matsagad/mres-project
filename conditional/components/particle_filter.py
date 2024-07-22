from enum import Enum
from protein.frames import Frames
from torch import Tensor
import torch
from typing import Callable, List


class LikelihoodMethod(str, Enum):
    MASK = "mask"
    MATRIX = "matrix"
    DISTANCE = "distance"


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
                x_t: Frames, y_t: Frames, y_mask: Tensor, variance: float
            ) -> Tensor:
                OBSERVED_REGION = y_mask[0] == 1

                centred_x_trans = x_t.trans[:, OBSERVED_REGION] - torch.mean(
                    x_t.trans[:, OBSERVED_REGION], dim=1
                ).unsqueeze(1)

                return -0.5 * (
                    ((centred_x_trans - y_t.trans[:, OBSERVED_REGION]) ** 2) / variance
                ).sum(dim=(1, 2))

            return log_likelihood

        if likelihood_method == LikelihoodMethod.MATRIX:

            def log_likelihood(
                x_t: Frames,
                y_t: Frames,
                y_mask: Tensor,
                A: Tensor,
                variance: float,
            ) -> Tensor:
                N_COORDS_PER_RESIDUE = 3
                OBSERVED_REGION = y_mask[0] == 1
                d, D = A.size()
                N_RESIDUES = D // N_COORDS_PER_RESIDUE

                return -0.5 * (
                    (
                        (
                            y_t.trans[:, OBSERVED_REGION].view(-1, d)
                            - x_t.trans[:, :N_RESIDUES].view(-1, D) @ A.T
                        )
                        ** 2
                    )
                    / variance
                ).sum(dim=1)

            return log_likelihood

        if likelihood_method == LikelihoodMethod.DISTANCE:

            def log_likelihood(
                x_t: Frames, y_t: Frames, y_mask: Tensor, variance: float
            ) -> Tensor:
                OBSERVED_REGION = y_mask[0] == 1
                N_OBSERVED = torch.sum(OBSERVED_REGION)
                _i, _j = torch.triu_indices(N_OBSERVED, N_OBSERVED, offset=1)

                dist_y = torch.cdist(
                    y_t.trans[:, OBSERVED_REGION], y_t.trans[:, OBSERVED_REGION]
                )[:, _i, _j].view(1, (N_OBSERVED * (N_OBSERVED - 1)) // 2)
                dist_x = torch.cdist(
                    x_t.trans[:, OBSERVED_REGION],
                    x_t.trans[:, OBSERVED_REGION],
                )[:, _i, _j].view(-1, (N_OBSERVED * (N_OBSERVED - 1)) // 2)

                return (-0.5 * ((dist_y - dist_x) ** 2) / variance).sum(dim=1)

            return log_likelihood

        raise KeyError(
            f"No such supported likelihood method: {self.likelihood_method}."
        )
