from abc import ABC, abstractmethod
from protein.frames import Frames
from pytorch_lightning.core import LightningModule
from torch import Tensor


# Interface adapted from genie/diffusion/diffusion.py
class FrameDiffusionModel(LightningModule, ABC):

    @abstractmethod
    def setup_schedule(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def transform(self, batch: Tensor) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def sample_timesteps(self, n_samples: int) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def sample_frames(self, mask: Tensor) -> Frames:
        raise NotImplementedError

    @abstractmethod
    def forward_diffuse(self, x_t: Frames, t: Tensor, mask: Tensor) -> Frames:
        raise NotImplementedError

    @abstractmethod
    def reverse_diffuse(
        self, x_t: Frames, t: Tensor, mask: Tensor, noise_scale: float
    ) -> Frames:
        raise NotImplementedError
