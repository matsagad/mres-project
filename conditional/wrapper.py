from abc import ABC, abstractmethod
from model.diffusion import FrameDiffusionModel
from torch import Tensor
import torch
import tqdm


class ConditionalWrapper(ABC):
    def __init__(self, model: FrameDiffusionModel) -> None:
        self.model = model

    @abstractmethod
    def sample_given_motif(self):
        """Sample conditioned on motif being present"""
        raise NotImplementedError

    def sample(self, mask: Tensor, noise_scale: float, verbose=True) -> Tensor:
        """Unconditional"""
        if not self.model.setup:
            self.setup_schedule()

        x_T = self.sample_frames(mask)
        x_trajectory = [x_T]
        x_t = x_T

        with torch.no_grad():
            for i in tqdm(
                reversed(range(self.model.n_timesteps)),
                desc="Reverse diffuse samples",
                total=self.model.n_timesteps,
                disable=not verbose,
            ):
                t = torch.tensor([i] * mask.shape[0], device=self.device).long()
                x_t = self.model.reverse_diffuse(x_t, t, mask, noise_scale)
                x_trajectory.append(x_t)

        return x_trajectory
