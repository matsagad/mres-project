from genie.genie.diffusion.genie import Genie
from model.diffusion import FrameDiffusionModel
from protein.frames import Frames
from torch import Tensor


class GenieAdapter(FrameDiffusionModel):
    def __init__(self, model: Genie) -> None:
        self.model = model

    def setup_schedule(self) -> None:
        self.model.setup_schedule()

    def transform(self, batch: Tensor) -> Tensor:
        return self.model.transform(batch)

    def sample_timesteps(self, n_samples: int) -> Tensor:
        return self.model.sample_timesteps(n_samples)

    def sample_frames(self, mask: Tensor) -> Frames:
        return self.model.sample_frames(mask)

    def forward_diffuse(self, x_t: Frames, t: Tensor, mask: Tensor) -> Frames:
        # No need to work with noise added
        x_t_plus_1, _ = self.model.q(x_t, t, mask)
        return x_t_plus_1

    def reverse_diffuse(
        self, x_t: Frames, t: Tensor, mask: Tensor, noise_scale: float
    ) -> Frames:
        return self.model.p(x_t, t, mask, noise_scale)
