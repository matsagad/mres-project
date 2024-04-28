from abc import ABC, abstractmethod
from model.diffusion import FrameDiffusionModel
import os
from torch import Tensor
import torch
import tqdm
from typing import Dict
from utils.path import out_dir


class ConditionalWrapper(ABC):
    def __init__(self, model: FrameDiffusionModel) -> None:
        self.model = model
        self.device = model.device
        self.verbose = True

    @property
    def device(self) -> str:
        return self._device

    @device.setter
    def device(self, _device: int) -> None:
        self._device = _device

    @property
    def verbose(self) -> bool:
        return self._verbose

    @verbose.setter
    def verbose(self, _verbose: bool) -> None:
        self._verbose = _verbose

    @abstractmethod
    def sample_given_motif(
        self, mask: Tensor, motif: Tensor, motif_mask: Tensor
    ) -> Tensor:
        """Sample conditioned on motif being present"""
        raise NotImplementedError

    def sample(self, mask: Tensor) -> Tensor:
        """Unconditional"""
        NOISE_SCALE = self.model.noise_scale

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
                disable=not self.verbose,
            ):
                t = torch.tensor([i] * mask.shape[0], device=self.device).long()
                x_t = self.model.reverse_diffuse(x_t, t, mask, noise_scale=NOISE_SCALE)
                x_trajectory.append(x_t)

        return x_trajectory

    def save_stats(self, stats: Dict[str, any]) -> None:
        out = out_dir()
        os.makedirs(os.path.join(out, "stats"), exist_ok=True)
        for stat, values in stats.items():
            if not values:
                continue
            tensor_values = (
                torch.stack(values)
                if type(values[0]) == torch.Tensor
                else torch.tensor(values)
            )
            torch.save(tensor_values, os.path.join(out, "stats", f"{stat}.pt"))
