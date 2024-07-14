from abc import ABC
from torch import Tensor
import torch
from typing import Callable, List


class ParticleFilter(ABC):

    @property
    def resample_indices(self) -> Callable[[Tensor], Tensor]:
        return self._resample_indices

    @resample_indices.setter
    def resample_indices(self, _resample_indices: Callable[[Tensor], Tensor]) -> None:
        self._resample_indices = _resample_indices

    def resample(self, w: Tensor, ess: Tensor, objects: List[Tensor]) -> None:
        n_batches, K_batch = w.shape
        K = n_batches * K_batch

        need_resampling = ess.cpu() <= 0.5 * K
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
