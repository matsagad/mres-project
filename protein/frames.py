from abc import ABC, abstractmethod
from torch import Tensor


class Frames(ABC):
    """Protein Backbone Frames"""

    @property
    @abstractmethod
    def trans(self) -> Tensor:
        """Translation vector"""
        pass

    @property
    @abstractmethod
    def rots(self) -> Tensor:
        """Rotation matrix"""
        pass
