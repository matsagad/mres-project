import re
from torch import Tensor
import torch


def pdb_to_c_alpha_backbone(f_pdb: str) -> Tensor:
    # Only extract C-alpha atom lines
    LABELS = "xyz"
    ca_line = re.compile(
        r"^ATOM\s*[0-9]+\s*CA\s*[a-zA-Z\s]+\s*[0-9]+\s*"
        + "\s+".join(f"(?P<{label}>[0-9\.\-]+)" for label in LABELS)
    )

    backbone = []
    with open(f_pdb, "r") as f:
        for line in f.readlines():
            match = ca_line.match(line)
            if not match:
                continue
            coords_dict = match.groupdict()
            backbone.append([float(coords_dict[label]) for label in LABELS])

    return torch.tensor(backbone)


def c_alpha_backbone_to_pdb(
    backbone: Tensor, f_out: str, placeholder_aa: str = "ALA A"
) -> None:
    LABELS = "xyz"
    N_DIGITS = 6
    temp = (
        "ATOM  {atom_idx:>5}  CA  {aa:>5} {res_idx:>3}"
        + " " * 5
        + " ".join(f"{{{label}:>{N_DIGITS}}}" for label in LABELS)
        + " " * 17
        + "C"
    )

    lines = []
    for i, coords in enumerate(backbone.squeeze()):
        trunc_coords = [str(coord.item())[: N_DIGITS + 1] for coord in coords]
        lines.append(
            temp.format(
                atom_idx=i,
                aa=placeholder_aa,
                res_idx=i,
                **dict(zip(LABELS, trunc_coords)),
            )
        )

    with open(f_out, "w") as f:
        f.write("\n".join(lines))
