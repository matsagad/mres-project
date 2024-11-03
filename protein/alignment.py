from torch import Tensor
import torch


def rmsd_kabsch(x_t: Tensor, y_t: Tensor) -> Tensor:
    """
    Kabsch algorithm for computing RMSD using SVD (Kabsch, 1976)
    """
    x = x_t - x_t.mean(dim=1, keepdim=True)
    y = y_t - y_t.mean(dim=1, keepdim=True)

    cov_mat = torch.einsum("ijk,ljm->ikm", x, y)

    U, _, V = torch.linalg.svd(cov_mat)
    is_negative = torch.det(U @ V) < 0

    V_no_reflections = V.clone()
    V_no_reflections[is_negative, :, -1] = -V[is_negative, :, -1]
    R = (U @ V_no_reflections).transpose(1, 2)

    rmsd = (((x @ R.transpose(1, 2)) - y) ** 2).sum(dim=2).mean(dim=1).sqrt()

    return rmsd


def _cross_cov_to_quaternion(m: Tensor) -> Tensor:

    return torch.stack(
        [
            torch.stack(arr)
            for arr in [
                [
                    m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2],
                    m[:, 1, 2] - m[:, 2, 1],
                    m[:, 2, 0] - m[:, 0, 2],
                    m[:, 0, 1] - m[:, 1, 0],
                ],
                [
                    m[:, 1, 2] - m[:, 2, 1],
                    m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2],
                    m[:, 0, 1] + m[:, 1, 0],
                    m[:, 0, 2] + m[:, 2, 0],
                ],
                [
                    m[:, 2, 0] - m[:, 0, 2],
                    m[:, 0, 1] + m[:, 1, 0],
                    -m[:, 0, 0] + m[:, 1, 1] - m[:, 2, 2],
                    m[:, 1, 2] + m[:, 2, 1],
                ],
                [
                    m[:, 0, 1] - m[:, 1, 0],
                    m[:, 0, 2] + m[:, 2, 0],
                    m[:, 1, 2] + m[:, 2, 1],
                    -m[:, 0, 0] - m[:, 1, 1] + m[:, 2, 2],
                ],
            ]
        ]
    ).moveaxis(-1, 0)


def rmsd_quaternion(x_t: Tensor, y_t: Tensor) -> Tensor:
    """
    Quaternions for computing RMSD (Coutsias, 2004)
    """
    x = x_t - x_t.mean(dim=1, keepdim=True)
    y = y_t - y_t.mean(dim=1, keepdim=True)

    cov_mat = torch.einsum("ijk,ljm->ikm", x, y)

    F = _cross_cov_to_quaternion(cov_mat)

    # Note: torch.linalg.eigvalsh returns eigenvalues in ascending order
    max_eigen_value = torch.linalg.eigvalsh(F)[:, -1]

    motif_len = y_t.shape[1]
    rmsd = torch.sqrt(
        torch.nn.functional.relu((x**2 + y**2).sum(dim=(1, 2)) - 2 * max_eigen_value)
        / motif_len
    )

    return rmsd
