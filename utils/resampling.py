import torch
from torch import Tensor
from typing import Callable, Tuple

"""
Resampling methods for particle filtering.

Each one below takes in weights of each particle
and returns the selected indices.
"""

RESAMPLING_REGISTRY = {}


def register_resampling_method(name: str):
    def register(resampling_method: Callable) -> Callable:
        if name in RESAMPLING_REGISTRY:
            raise Exception(f"Resampling method '{name}' already registered!")
        RESAMPLING_REGISTRY[name] = resampling_method
        return resampling_method

    return register


def get_resampling_method(method: str) -> Callable:
    if method not in RESAMPLING_REGISTRY:
        raise Exception(
            f"Invalid resampling method '{method}'. "
            f"Choose from: {', '.join(RESAMPLING_REGISTRY.keys())}."
        )
    return RESAMPLING_REGISTRY[method]


@register_resampling_method("residual")
def residual_resample(w: Tensor) -> Tensor:
    squeeze = False
    if w.ndim == 1:
        w = w.view(1, -1)
        squeeze = True

    assert w.ndim == 2

    N, K = w.shape
    c_k = torch.floor(K * w).int()
    r_k = K * w - c_k
    R = K - torch.sum(c_k, axis=1)

    indices = []
    for i in range(N):
        i_C = torch.repeat_interleave(torch.arange(K).to(w.device), c_k[i])
        if R[i] == 0:
            indices.append(i_C)
            continue
        i_R = torch.multinomial(r_k[i], R[i], replacement=True)
        indices.append(torch.cat((i_R, i_C)))
    indices = torch.stack(indices)

    if squeeze:
        return indices.squeeze(0)
    return indices


@register_resampling_method("stratified")
def stratified_resample(w: Tensor) -> Tensor:
    squeeze = False
    if w.ndim == 1:
        w = w.view(1, -1)
        squeeze = True

    assert w.ndim == 2

    N, K = w.shape
    w_cumsum = torch.cumsum(w, 1)
    samples = (torch.rand((N, K)) + torch.arange(K).float().view(1, K)) / K
    indices = torch.searchsorted(w_cumsum, samples.to(w.device))

    if squeeze:
        return indices.squeeze(0)
    return indices


@register_resampling_method("systematic")
def systematic_resample(w: Tensor) -> Tensor:
    squeeze = False
    if w.ndim == 1:
        w = w.view(1, -1)
        squeeze = True

    assert w.ndim == 2

    N, K = w.shape
    w_cumsum = torch.cumsum(w, 1)
    samples = (
        torch.rand(N).repeat_interleave(K).view(N, K)
        + torch.arange(K).float().view(1, K)
    ) / K
    indices = torch.searchsorted(w_cumsum, samples.to(w.device))

    if squeeze:
        return indices.squeeze(0)
    return indices


def get_unique_and_inverse(x: Tensor) -> Tuple[Tensor, Tensor]:
    # Adapted from https://github.com/pytorch/pytorch/issues/36748
    unique_values, inverse = torch.unique(x, dim=0, return_inverse=True)
    perm = torch.arange(inverse.shape[0], device=inverse.device)
    unique = inverse.new_empty(unique_values.shape[0]).scatter_(0, inverse, perm)
    return unique, inverse
