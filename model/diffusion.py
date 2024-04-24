from abc import ABC, abstractmethod
from protein.frames import Frames
from pytorch_lightning.core import LightningModule
from torch import Tensor


# Interface adapted from genie/diffusion/diffusion.py
class FrameDiffusionModel(LightningModule, ABC):

    @property
    def batch_size(self) -> int:
        return 1

    @property
    def n_timesteps(self) -> int:
        raise NotImplementedError

    @property
    def setup(self) -> bool:
        return False

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
    def forward_log_likelihood(
        self,
        x_t_plus_one: Frames,
        x_t: Frames,
        t: Tensor,
        llik_mask: Tensor,
        mask: Tensor,
    ) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def reverse_diffuse(
        self, x_t: Frames, t: Tensor, mask: Tensor, noise_scale: float = 1.0
    ) -> Frames:
        raise NotImplementedError

    @abstractmethod
    def reverse_log_likelihood(
        self,
        x_t_minus_one: Frames,
        x_t: Frames,
        t: Tensor,
        llik_mask: Tensor,
        mask: Tensor,
    ) -> Tensor:
        raise NotImplementedError
