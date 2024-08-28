from abc import ABC, abstractmethod
from torch import Tensor
import torch


class Frames(ABC):
    """Protein Backbone Frames"""

    @property
    @abstractmethod
    def rots(self) -> Tensor:
        """Rotation matrix"""
        pass

    @property
    @abstractmethod
    def trans(self) -> Tensor:
        """Translation vector"""
        pass

    @abstractmethod
    def __init__(self, rots: Tensor, trans: Tensor):
        self.rots = rots
        self.trans = trans


def compute_frenet_frames(x: Tensor, mask: Tensor, eps: float = 1e-10) -> Tensor:
    """
    Logic for computing frames given the C-alpha coordinates x.
    (Copied over from the Genie repository but modified to be stable
     with autograd by avoiding setting views of tbn to _rots)

    TODO: check that no more 'RuntimeError: _Map_base::at' for long proteins
    """
    t = x[:, 1:] - x[:, :-1]
    t_norm = torch.sqrt(eps + torch.sum(t**2, dim=-1))
    t = t / t_norm.unsqueeze(-1)

    b = torch.cross(t[:, :-1], t[:, 1:])
    b_norm = torch.sqrt(eps + torch.sum(b**2, dim=-1))
    b = b / b_norm.unsqueeze(-1)

    n = torch.cross(b, t[:, 1:])
    tbn = torch.stack([t[:, 1:], b, n], dim=-1)

    rots = []
    for i in range(mask.shape[0]):
        length = torch.sum(mask[i]).int()

        _rots = torch.cat(
            [
                tbn[i, :1].clone(),
                tbn[i, : length - 2],
                tbn[i, length - 3 : length - 2].clone(),
                torch.eye(3, device=x.device)
                .unsqueeze(0)
                .repeat(mask.shape[1] - length, 1, 1),
            ]
        )
        rots.append(_rots)
    rots = torch.stack(rots, dim=0).to(x.device)

    return rots
