import torch
from torch import Tensor
from typing import Callable

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
    K = w.shape[0]
    c_k = torch.floor(K * w).int()
    r_k = K * w - c_k

    i_C = torch.repeat_interleave(torch.arange(K).to(w.device), c_k)

    R = K - torch.sum(c_k, axis=0)
    if R == 0:
        return i_C
    i_R = torch.multinomial(r_k, R)
    return torch.cat((i_R, i_C))


@register_resampling_method("stratified")
def stratified_resample(w: Tensor) -> Tensor:
    K = w.shape[0]
    w_cumsum = torch.cumsum(w, 0)
    samples = (torch.rand(K) + torch.arange(K).float()) / K
    return torch.searchsorted(w_cumsum, samples.to(w.device))


@register_resampling_method("systematic")
def systematic_resample(w: Tensor) -> Tensor:
    K = w.shape[0]
    w_cumsum = torch.cumsum(w, 0)
    samples = (torch.rand(1) + torch.arange(K).float()) / K
    return torch.searchsorted(w_cumsum, samples.to(w.device))
