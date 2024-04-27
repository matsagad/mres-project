import torch
from torch import Tensor

"""
Resampling methods for particle filtering.

Each one below takes in weights of each particle
and returns the selected indices.
"""


def residual_resample(w: Tensor) -> Tensor:
    K = w.shape[0]
    c_k = torch.floor(K * w).int()
    r_k = K * w - c_k

    i_C = torch.repeat_interleave(torch.arange(K).to(w.device), c_k)

    R = K - torch.sum(c_k, axis=0)
    if R == 0:
        return i_C
    i_R = torch.multinomial(r_k, R)
    return torch.cat((i_R, i_C))


def stratified_resample(w):
    K = w.shape[0]
    w_cumsum = torch.cumsum(w, 0)
    samples = (torch.rand(K) + torch.arange(K).float()) / K
    return torch.searchsorted(w_cumsum, samples.to(w.device))


def systematic_resample(w):
    K = w.shape[0]
    w_cumsum = torch.cumsum(w, 0)
    samples = (torch.rand(1) + torch.arange(K).float()) / K
    return torch.searchsorted(w_cumsum, samples.to(w.device))

RESAMPLING_METHOD = {
    "residual": residual_resample,
    "stratified": stratified_resample,
    "systematic": systematic_resample,
}