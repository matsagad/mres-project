import re
from torch import Tensor
import torch
from typing import Tuple


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


def get_motif_mask(
    motif: Tensor, n_residues: int, max_n_residues: int, contig: str = None
) -> Tuple[Tensor, Tensor]:
    COMMA = ","
    DASH = "-"
    OFFSET = 1

    motif_trans = torch.zeros((1, max_n_residues, 3), device=motif.device)
    motif_mask = torch.zeros((1, max_n_residues), device=motif.device)

    if not contig:
        motif_trans[:, : motif.shape[0]] = motif - torch.mean(
            motif, dim=0, keepdim=True
        )
        motif_mask[:, : motif.shape[0]] = 1
        return motif_trans, motif_mask

    trans_locations = []
    motif_locations = []

    _n_motif_residues = 0
    _total = torch.zeros(3, device=motif.device)

    curr = 0
    for _range in contig.split(COMMA):
        if _range[0].isalpha():
            if DASH in _range[1:]:
                motif_start, motif_end = map(int, _range[1:].split(DASH))
                motif_end += 1
            else:
                motif_start = int(_range[1:])
                motif_end = motif_start + 1

            motif_mask[:, curr : curr + motif_end - motif_start] = 1

            trans_locations.append((curr, curr + motif_end - motif_start))
            motif_locations.append(motif[motif_start - OFFSET : motif_end - OFFSET])

            _total += torch.sum(motif_locations[-1], dim=0)
            _n_motif_residues += motif_end - motif_start
            curr += motif_end - motif_start
        else:
            if DASH in _range:
                min_scaffold_segment, _ = map(int, _range.split(DASH))
            else:
                min_scaffold_segment = int(_range)

            curr += min_scaffold_segment

        assert (
            curr + 1 <= n_residues
        ), f"Exceeded experiment's sample length: ({curr + 1} > {n_residues})"

    motif_mean = _total / _n_motif_residues

    for (start, end), motif_loc in zip(trans_locations, motif_locations):
        motif_trans[:, start:end] = motif_loc - motif_mean
    return motif_trans, motif_mask
