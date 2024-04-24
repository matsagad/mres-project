from abc import ABC, abstractmethod
from torch import Tensor


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
