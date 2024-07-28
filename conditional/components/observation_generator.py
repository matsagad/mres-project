from enum import Enum
from model.diffusion import FrameDiffusionModel
from protein.frames import Frames
import torch
from torch import Tensor
from tqdm import tqdm
from typing import List


class ObservationGenerationMethod(str, Enum):
    BACKWARD = "backward"
    FORWARD = "forward"


class LinearObservationGenerator:
    """
    Generate sequence of observations {y_t}, for y_0 = A @ x_0.

    1. Forward. As in SMCDiff, we forward-diffuse the observation
       according to the forward process.
    2. Backward. As in FPSSMC, we perform a noise-sharing technique
       by building a recurrence relation for the observed sequence
       that mirrors that of the latent variables x_t.
    """

    @property
    def model(self) -> FrameDiffusionModel:
        return self._model

    @model.setter
    def model(self, _model: FrameDiffusionModel) -> None:
        self._model = _model

    @property
    def observed_sequence_method(self) -> ObservationGenerationMethod:
        return self._observed_sequence_method

    @observed_sequence_method.setter
    def observed_sequence_method(
        self, _observed_sequence_method: ObservationGenerationMethod
    ) -> None:
        self._observed_sequence_method = _observed_sequence_method

    @property
    def observed_sequence_noised(self) -> bool:
        return self._observed_sequence_noised

    @observed_sequence_noised.setter
    def observed_sequence_noised(self, _observed_sequence_noised: bool) -> None:
        self._observed_sequence_noised = _observed_sequence_noised

    def generate_observed_sequence(
        self,
        mask: Tensor,
        y_zero: Frames,
        y_mask: Tensor,
        A: List[Tensor],
        x_T: Frames = None,
        recenter_y: bool = True,
    ) -> List[Tensor]:
        N_TIMESTEPS = self.model.n_timesteps
        N_COORDS_PER_RESIDUE = 3
        N_RESIDUES = (mask[0] == 1).sum().item()
        N_OBSERVATIONS = y_mask.shape[0]

        if self.observed_sequence_method == ObservationGenerationMethod.FORWARD:
            # Construct y sequence by forward diffusing the final observation y_0
            y_t = y_zero
            if recenter_y:
                for j in range(N_OBSERVATIONS):
                    y_t.trans[j, y_mask[j] == 1] -= torch.mean(
                        y_t.trans[j, y_mask[j] == 1], dim=0, keepdim=True
                    )
            y_sequence = [y_t]
            for i in tqdm(
                range(N_TIMESTEPS),
                desc="Generating {y_t}",
                total=N_TIMESTEPS,
                disable=not self.verbose,
            ):
                t = torch.tensor([i], device=self.device).long()
                sqrt_alpha_t = torch.sqrt(1 - self.model.variance[t])
                sqrt_one_minus_alpha_t = self.model.sqrt_variance[t]

                y_t_trans = torch.zeros(y_t.trans.shape, device=self.device)
                y_t_trans[y_mask == 1] = sqrt_alpha_t * y_t.trans[y_mask == 1]

                if self.observed_sequence_noised:
                    y_t_trans[y_mask == 1] += sqrt_one_minus_alpha_t * torch.randn(
                        y_t_trans[y_mask == 1].shape, device=self.device
                    )
                if recenter_y:
                    for j in range(N_OBSERVATIONS):
                        y_t_trans[j, y_mask[j] == 1] -= torch.mean(
                            y_t_trans[j, y_mask[j] == 1], dim=0, keepdim=True
                        )
                y_t = self.model.coords_to_frames(y_t_trans, y_mask)

                y_sequence.append(y_t)

            return y_sequence

        if self.observed_sequence_method == ObservationGenerationMethod.BACKWARD:
            # Construct the y sequence backwards by recursively interpolating
            # with the motif and matching the reverse-process for x
            y_T_trans = torch.zeros(y_zero.trans.shape, device=self.device)
            for j in range(len(y_mask)):
                y_T_trans[j, y_mask[j] == 1] = (
                    x_T.trans[:1, :N_RESIDUES].flatten() @ A[j].T
                ).view(1, -1, N_COORDS_PER_RESIDUE)
            y_T = self.model.coords_to_frames(y_T_trans, y_mask)

            y_t = y_T
            y_sequence = [y_T]

            for i in tqdm(
                reversed(range(N_TIMESTEPS)),
                desc="Generating {y_t}",
                total=N_TIMESTEPS,
                disable=not self.verbose,
            ):
                t = torch.tensor([i], device=self.device).long()

                alpha_bar_t = 1 - self.model.forward_variance[t]
                alpha_bar_t_minus_one = 1 - (
                    self.model.forward_variance[t - 1] if t[0] > 0 else torch.tensor(0)
                )

                c = self.model.variance[t] / (1 - alpha_bar_t)
                p_t = torch.sqrt(
                    (1 - c) * (1 - alpha_bar_t_minus_one) / (1 - alpha_bar_t)
                )
                q_t = torch.sqrt(c * (1 - alpha_bar_t_minus_one))

                y_t_minus_one_trans = torch.zeros(
                    y_zero.trans.shape, device=self.device
                )
                y_t_minus_one_trans[y_mask == 1] = torch.sqrt(
                    alpha_bar_t_minus_one
                ) * y_zero.trans[y_mask == 1] + p_t * (
                    y_t.trans[y_mask == 1]
                    - torch.sqrt(alpha_bar_t) * y_zero.trans[y_mask == 1]
                )

                if self.observed_sequence_noised:
                    for j in range(N_OBSERVATIONS):
                        y_t_minus_one_trans[j, y_mask[j] == 1] += q_t * (
                            A[j] @ torch.randn((A[j].shape[1],), device=self.device)
                        ).view(-1, N_COORDS_PER_RESIDUE)
                if recenter_y:
                    for j in range(N_OBSERVATIONS):
                        y_t_minus_one_trans[j, y_mask[j] == 1] -= torch.mean(
                            y_t_minus_one_trans[j, y_mask[j] == 1], dim=0, keepdim=True
                        )
                y_t_minus_one = self.model.coords_to_frames(y_t_minus_one_trans, y_mask)

                y_t = y_t_minus_one
                y_sequence.append(y_t)

            y_sequence.append(y_zero)
            y_sequence = y_sequence[::-1]

            return y_sequence
