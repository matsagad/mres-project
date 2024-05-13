from genie.config import Config
from genie.diffusion.genie import Genie
from genie.utils.geo_utils import compute_frenet_frames
from genie.utils.affine_utils import T
from model.diffusion import FrameDiffusionModel
from protein.frames import Frames
import torch
from torch import Tensor
from typing import Union


class GenieAdapter(FrameDiffusionModel):
    def __init__(self, model: Genie) -> None:
        super().__init__()
        self.model = model

        self.batch_size = 5
        self.n_timesteps = model.config.diffusion["n_timestep"]
        self.max_n_residues = model.config.io["max_n_res"]

        self.setup_schedule()

        self.variance = self.model.betas
        self.sqrt_variance = self.model.sqrt_betas
        self.forward_variance = self.model.one_minus_alphas_cumprod
        self.sqrt_forward_variance = self.model.sqrt_one_minus_alphas_cumprod

    def from_weights_and_config(f_weights: str, f_config: str) -> "GenieAdapter":
        config = Config(f_config)
        model = Genie.load_from_checkpoint(f_weights, config=config)
        model.eval()
        model.model.eval()
        return GenieAdapter(model)

    def setup_schedule(self) -> None:
        self.model.setup_schedule()
        self.setup = True

    def _setup_on_device(self) -> None:
        self_attrs = [
            "variance",
            "sqrt_variance",
            "forward_variance",
            "sqrt_forward_variance",
        ]
        for attr in self_attrs:
            setattr(self, attr, getattr(self, attr).to(self.device))

        model_attrs = [
            "alphas",
            "sqrt_alphas",
            "sqrt_alphas_cumprod",
            "one_minus_alphas_cumprod",
            "sqrt_one_minus_alphas_cumprod",
            "betas",
            "sqrt_betas",
        ]
        for attr in model_attrs:
            setattr(self.model, attr, getattr(self.model, attr).to(self.device))

    def transform(self, batch: Tensor) -> Tensor:
        return self.model.transform(batch)

    def sample_timesteps(self, n_samples: int) -> Tensor:
        return self.model.sample_timesteps(n_samples)

    def sample_frames(self, mask: Tensor) -> Frames:
        return self.model.sample_frames(mask)

    def coords_to_frames(self, coords: Tensor, mask: Tensor) -> Frames:
        rots = compute_frenet_frames(coords, mask)
        return T(rots, coords)

    def forward_diffuse(self, x_t: Frames, t: Tensor, mask: Tensor) -> Frames:
        # No need to work with noise added
        x_t_plus_one, _ = self.model.q(x_t, t, mask)
        return x_t_plus_one

    def forward_diffuse_deterministic(
        self, x_t: Frames, t: Tensor, mask: Tensor
    ) -> Frames:
        x_t_plus_one_trans = self.model.sqrt_alphas[t].view(-1, 1, 1) * (x_t.trans)
        x_t_plus_one = self.coords_to_frames(x_t_plus_one_trans, mask)
        return x_t_plus_one

    def forward_log_likelihood(
        self,
        x_t_plus_one: Frames,
        x_t: Frames,
        t: Tensor,
        llik_mask: Tensor,
        mask: Tensor,
    ) -> Tensor:
        # Find probability density
        mu = self.model.sqrt_alphas[t].view(-1, 1, 1) * (x_t.trans)
        sigma = self.model.sqrt_betas[t].view(-1, 1, 1)

        log_density = (
            -0.5 * ((x_t_plus_one.trans - mu)[:, llik_mask[0] == 1] / sigma) ** 2
        ).sum(axis=(1, 2)) - 0.5 * torch.log(2 * torch.pi * sigma * sigma)

        return log_density

    def reverse_diffuse(
        self, x_t: Frames, t: Tensor, mask: Tensor, noise_scale: float
    ) -> Frames:
        return self.model.p(x_t, t, mask, noise_scale)

    def reverse_diffuse_deterministic(
        self, x_t: Frames, t: Tensor, mask: Tensor
    ) -> Frames:
        x_t_minus_one_trans = (
            x_t.trans + self.model.betas[t].view(-1, 1, 1) * self.score(x_t, t, mask)
        ) / self.model.sqrt_alphas[t].view(-1, 1, 1)

        x_t_minus_one = self.coords_to_frames(x_t_minus_one_trans, mask)
        return x_t_minus_one

    def _epsilon(self, x_t: Frames, t: Tensor, mask: Tensor) -> Tensor:
        denoised_pile = []
        for batch in torch.split(torch.arange(x_t.shape[0]), self.batch_size):
            curr_batch_size = len(batch)
            denoised_trans = self.model.model(
                T(x_t.rots[batch], x_t.trans[batch]),
                t[:curr_batch_size],
                mask[:curr_batch_size],
            ).trans.detach()
            denoised_pile.append(denoised_trans)

        return x_t.trans - torch.cat(denoised_pile, dim=0)

    def predict_fully_denoised(self, x_t: Frames, t: Tensor, mask: Tensor) -> Frames:
        epsilon = self._epsilon(x_t, t, mask)

        x_0_trans = (
            x_t.trans
            - self.model.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1) * epsilon
        ) / self.model.sqrt_alphas_cumprod[t].view(-1, 1, 1)

        return self.coords_to_frames(x_0_trans, mask)

    def reverse_log_likelihood(
        self,
        x_t_minus_one: Frames,
        x_t: Frames,
        t: Tensor,
        llik_mask: Tensor,
        mask: Tensor,
    ) -> Tensor:
        # Find noise prediction
        ## [B, N_AA, 3]
        epsilon = self._epsilon(x_t, t, mask)

        # Find probability density
        noise_scale = (
            self.model.betas[t] / self.model.sqrt_one_minus_alphas_cumprod[t]
        ).view(-1, 1, 1)
        mu = (1.0 / self.model.sqrt_alphas[t]).view(-1, 1, 1) * (
            x_t.trans - noise_scale * epsilon
        )
        sigma_squared = self.model.betas[t]

        ## We can skip the normalising constant for most use-cases, e.g. for
        ## importance weights calculation, but keep it for completeness
        log_density = (
            -0.5
            * ((x_t_minus_one.trans - mu)[:, llik_mask[0] == 1] ** 2)
            / sigma_squared.view(-1, 1, 1)
        ).sum(axis=(1, 2)) - 0.5 * torch.log(2 * torch.pi * sigma_squared)

        return log_density

    def score(self, x_t: Frames, t: Tensor, mask: Tensor) -> Tensor:
        epsilon = self._epsilon(x_t, t, mask)
        return -epsilon / self.model.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)

    def to(self, device: Union[str, torch.device]) -> "GenieAdapter":
        self.model = self.model.to(device)
        self.device = device
        self._setup_on_device()
        return super().to(device)
