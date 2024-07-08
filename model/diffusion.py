from abc import ABC, abstractmethod
from contextlib import contextmanager
from protein.frames import Frames
from pytorch_lightning.core import LightningModule
from torch import Tensor


# Future TODO: make less restrictive to non-DDPM diffusion models
class FrameDiffusionModel(LightningModule, ABC):

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, _batch_size: int) -> None:
        self._batch_size = _batch_size

    @property
    def device(self) -> str:
        return self._device

    @device.setter
    def device(self, _device: str) -> None:
        self._device = _device

    @property
    def noise_scale(self) -> float:
        return self._noise_scale

    @noise_scale.setter
    def noise_scale(self, _noise_scale: float) -> None:
        self._noise_scale = _noise_scale

    @property
    def n_timesteps(self) -> int:
        return self._n_timesteps

    @n_timesteps.setter
    def n_timesteps(self, _n_timesteps: int) -> None:
        self._n_timesteps = _n_timesteps

    @property
    def max_n_residues(self) -> int:
        return self._max_n_residues

    @max_n_residues.setter
    def max_n_residues(self, _max_n_residues: int) -> None:
        self._max_n_residues = _max_n_residues

    @property
    def setup(self) -> bool:
        return self._setup

    @setup.setter
    def setup(self, _setup: bool) -> None:
        self._setup = _setup

    @property
    def variance(self) -> Tensor:
        return self._variance

    @variance.setter
    def variance(self, _variance: Tensor) -> None:
        self._variance = _variance

    @property
    def sqrt_variance(self) -> Tensor:
        return self._sqrt_variance

    @sqrt_variance.setter
    def sqrt_variance(self, _sqrt_variance: Tensor) -> None:
        self._sqrt_variance = _sqrt_variance

    @property
    def forward_variance(self) -> Tensor:
        return self._forward_variance

    @forward_variance.setter
    def forward_variance(self, _forward_variance: Tensor) -> None:
        self._forward_variance = _forward_variance

    @property
    def sqrt_forward_variance(self) -> Tensor:
        return self._sqrt_forward_variance

    @sqrt_forward_variance.setter
    def sqrt_forward_variance(self, _sqrt_forward_variance: Tensor) -> None:
        self._sqrt_forward_variance = _sqrt_forward_variance

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
    def coords_to_frames(self, coords: Tensor, mask: Tensor) -> Frames:
        raise NotImplementedError

    @abstractmethod
    def forward_diffuse(self, x_t: Frames, t: Tensor, mask: Tensor) -> Frames:
        raise NotImplementedError

    @abstractmethod
    def forward_diffuse_deterministic(
        self, x_t: Frames, t: Tensor, mask: Tensor
    ) -> Frames:
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
    def reverse_diffuse_deterministic(
        self, x_t: Frames, t: Tensor, mask: Tensor
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

    @abstractmethod
    def predict_fully_denoised(self, x_t: Frames, t: Tensor, mask: Tensor) -> Frames:
        raise NotImplementedError

    @abstractmethod
    def score(self, x_t: Frames, t: Tensor, mask: Tensor) -> Tensor:
        raise NotImplementedError

    def with_batch_size(self, batch_size: int) -> "FrameDiffusionModel":
        self.batch_size = batch_size
        return self

    def with_noise_scale(self, noise_scale: float) -> "FrameDiffusionModel":
        self.noise_scale = noise_scale
        return self

    @property
    def cached_epsilon(self) -> Tensor:
        return self._cached_epsilon

    @cached_epsilon.setter
    def cached_epsilon(self, _cached_epsilon: Tensor) -> None:
        self._cached_epsilon = _cached_epsilon

    @property
    def cached_score(self) -> Tensor:
        return self._cached_score

    @cached_score.setter
    def cached_score(self, _cached_score: Tensor) -> None:
        self._cached_score = _cached_score

    @contextmanager
    def with_epsilon(self, epsilon: Tensor) -> None:
        try:
            assert (
                self._cached_epsilon is None
            ), "Either context manager was nested or property _cached_epsilon was already set."
            self._cached_epsilon = epsilon
            yield
        finally:
            self._cached_epsilon = None

    @contextmanager
    def with_score(self, score: Tensor) -> None:
        try:
            assert (
                self._cached_score is None
            ), "Either context manager was nested or property _cached_score was already set."
            self._cached_score = score
            yield
        finally:
            self._cached_score = None
