import re
from torch import Tensor
import torch
from typing import Dict, List, Tuple, Union


def pdb_to_atom_backbone(f_pdb: str) -> Tensor:
    LABELS = "xyz"
    atoms = ["N", "CA", "C", "O"]
    atom_line = re.compile(
        rf"^ATOM\s*[0-9]+\s*(?P<atom>({'|'.join(atoms)}))\s+[a-zA-Z\s]+\s*[0-9]+\s*"
        + "\s+".join(f"(?P<{label}>[0-9\.\-]+)" for label in LABELS)
    )

    backbones = {atom: [] for atom in atoms}
    with open(f_pdb, "r") as f:
        for line in f.readlines():
            match = atom_line.match(line)
            if not match:
                continue
            coords_dict = match.groupdict()
            backbones[coords_dict["atom"]].append(
                [float(coords_dict[label]) for label in LABELS]
            )

    for atom, backbone in backbones.items():
        backbones[atom] = torch.tensor(backbone)

    return backbones


def pdb_to_c_alpha_backbone(f_pdb: str) -> Tensor:
    return pdb_to_atom_backbone(f_pdb)["CA"]


def atom_backbone_to_pdb(
    backbones: Dict[str, Tensor],
    f_out: str,
    placeholder_aa: str = "ALA A",
    atoms: List[str] = ["N", "CA", "C", "O"],
) -> None:
    LABELS = "xyz"
    N_DIGITS = 6
    temp = (
        "ATOM  {atom_idx:>5}  {atom:>2}  {aa:>5} {res_idx:>3}"
        + " " * 5
        + " ".join(f"{{{label}:>{N_DIGITS}}}" for label in LABELS)
        + " " * 17
        + "{atom_el:>2}"
    )
    squeezed_backbones = {
        atom: backbone.squeeze() for atom, backbone in backbones.items()
    }

    lines = []
    atom_i = 0
    for i in range(max(map(len, squeezed_backbones.values()))):
        for atom in atoms:
            if i >= len(squeezed_backbones[atom]):
                continue
            trunc_coords = [
                str(coord.item())[: N_DIGITS + 1]
                for coord in squeezed_backbones[atom][i]
            ]
            lines.append(
                temp.format(
                    atom_idx=atom_i,
                    atom=atom,
                    aa=placeholder_aa,
                    res_idx=i,
                    **dict(zip(LABELS, trunc_coords)),
                    atom_el=atom[0],
                )
            )
            atom_i += 1

    with open(f_out, "w") as f:
        f.write("\n".join(lines))


def c_alpha_backbone_to_pdb(
    backbone: Tensor, f_out: str, placeholder_aa: str = "ALA A"
) -> None:
    atom_backbone_to_pdb(
        {"CA": backbone.squeeze()}, f_out, placeholder_aa, atoms=["CA"]
    )


def get_motif_mask(
    motif_backbones: Dict[str, Tensor],
    max_n_residues: int,
    contig: str = None,
    return_masked_backbones: bool = False,
) -> Union[Tuple[Tensor, Tensor, Dict[str, Tensor]], Tuple[Tensor, Tensor]]:
    COMMA = ","
    DASH = "-"
    OFFSET = 1
    ATOM_TO_CENTER = "CA"
    ATOM_TO_MASK = "CA"

    out = {atom: torch.zeros((1, max_n_residues, 3)) for atom in motif_backbones.keys()}
    motif_mask = torch.zeros((1, max_n_residues))
    mask = torch.zeros((1, max_n_residues))

    if not contig:
        # If not specified, put motif at the start and generate protein with
        # total length equal to the maximum allowed.
        motif_mask[:, : motif_backbones[ATOM_TO_MASK].shape[0]] = 1
        mask[:, :max_n_residues] = 1
        if not return_masked_backbones:
            return motif_mask, mask
        for atom, backbone in motif_backbones.items():
            out[atom][:, : backbone.shape[0]] = backbone - torch.mean(
                motif_backbones[ATOM_TO_CENTER], dim=0, keepdim=True
            )
        return motif_mask, mask, out

    trans_locations = []
    motif_locations = {atom: [] for atom in motif_backbones.keys()}

    _n_motif_residues = 0
    _total = torch.zeros(3)

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
            for atom, atom_locations in motif_locations.items():
                atom_locations.append(
                    motif_backbones[atom][motif_start - OFFSET : motif_end - OFFSET]
                )

            _total += torch.sum(motif_locations[ATOM_TO_CENTER][-1], dim=0)
            _n_motif_residues += motif_end - motif_start
            curr += motif_end - motif_start
        else:
            if DASH in _range:
                min_segment_length, max_segment_length = map(int, _range.split(DASH))
                # Use median scaffold length similar to TDS paper
                scaffold_segment_length = (min_segment_length + max_segment_length) // 2
            else:
                scaffold_segment_length = int(_range)

            curr += scaffold_segment_length

        assert (
            curr + 1 <= max_n_residues
        ), f"Exceeded experiment's maximum number of residues: ({curr + 1} > {max_n_residues})"

    mask[:, : curr + 1] = 1
    if not return_masked_backbones:
        return motif_mask, mask

    motif_mean = _total / _n_motif_residues

    for i, (start, end) in enumerate(trans_locations):
        for atom, backbone in motif_backbones.items():
            out[atom][:, start:end] = motif_locations[atom][i] - motif_mean

    return motif_mask, mask, out
