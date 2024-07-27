import re
from torch import Tensor
import torch
from typing import Dict, List, Tuple, Union


def pdb_to_atom_backbone(
    f_pdb: str, atoms: List[str] = ["N", "CA", "C", "O"]
) -> Dict[str, Dict[str, Tensor]]:
    LABELS = "xyz"
    atom_line = re.compile(
        rf"^ATOM\s*[0-9]+\s*(?P<atom>({'|'.join(atoms)}))"
        + r"\s+[a-zA-Z]+\s+(?P<chain>[a-zA-Z])\s*(?P<chain_index>[0-9]+)\s*"
        + r"\s+".join(rf"(?P<{label}>[0-9\.\-]+)" for label in LABELS)
    )

    backbones = {}
    with open(f_pdb, "r") as f:
        for line in f.readlines():
            match = atom_line.match(line)
            if not match:
                continue
            coords_dict = match.groupdict()
            chain = coords_dict["chain"]

            if chain not in backbones:
                backbones[chain] = {atom: [] for atom in atoms}

            # Chain index may not always start at one, so we can
            # pad it with zeros for consistent indexing later
            diff = (
                int(coords_dict["chain_index"])
                - len(backbones[chain][coords_dict["atom"]])
                - 1
            )
            backbones[chain][coords_dict["atom"]].extend(
                [0.0, 0.0, 0.0] for _ in range(diff)
            )
            backbones[chain][coords_dict["atom"]].append(
                [float(coords_dict[label]) for label in LABELS]
            )

    for chain_backbones in backbones.values():
        for atom, backbone in chain_backbones.items():
            chain_backbones[atom] = torch.tensor(backbone)

    return backbones


def pdb_to_c_alpha_backbone(f_pdb: str) -> Tensor:
    return pdb_to_atom_backbone(f_pdb)["CA"]


def atom_backbone_to_pdb(
    backbones: Dict[str, Tensor],
    f_out: str,
    placeholder_aa: str = "ALA",
    placeholder_chain: str = "A",
    atoms: List[str] = ["N", "CA", "C", "O"],
) -> None:
    LABELS = "xyz"
    N_DIGITS = 6
    temp = (
        "ATOM  {atom_idx:>5}  {atom:>2}  {aa:>3} {chain:>1} {res_idx:>3}"
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
                    chain=placeholder_chain,
                    res_idx=i,
                    **dict(zip(LABELS, trunc_coords)),
                    atom_el=atom[0],
                )
            )
            atom_i += 1

    with open(f_out, "w") as f:
        f.write("\n".join(lines))


def c_alpha_backbone_to_pdb(
    backbone: Tensor,
    f_out: str,
    placeholder_aa: str = "ALA",
    placeholder_chain: str = "A",
) -> None:
    atom_backbone_to_pdb(
        {"CA": backbone.squeeze()},
        f_out,
        placeholder_aa,
        placeholder_chain,
        atoms=["CA"],
    )


def get_motif_mask(
    motif_backbones: Dict[str, Dict[str, Tensor]],
    max_n_residues: int,
    contig: str = None,
    return_masked_backbones: bool = False,
) -> Union[Tuple[Tensor, Tensor, Dict[str, Tensor]], Tuple[Tensor, Tensor]]:
    COMMA = ","
    DASH = "-"
    OFFSET = 1
    ATOM_TO_CENTER = "CA"
    ATOMS = list(list(motif_backbones.values())[0].keys())

    motif_mask = torch.zeros((1, max_n_residues))
    mask = torch.zeros((1, max_n_residues))

    trans_locations = []
    motif_locations = {atom: [] for atom in ATOMS}

    n_motif_residues = 0
    sum_of_coords = torch.zeros(3)

    chain_segment = re.compile(
        r"(?P<chain>[a-zA-Z]?)(?P<range>([0-9]+)|([0-9]+\-[0-9]+))"
    )
    curr_pos = 0
    for chunk in contig.split(COMMA):
        chunk_dict = chain_segment.fullmatch(chunk).groupdict()
        chunk_chain = chunk_dict["chain"]
        chunk_range = chunk_dict["range"]

        if chunk_dict["chain"]:
            if DASH in chunk_range:
                motif_start, motif_end = map(int, chunk_range.split(DASH))
                motif_end += 1
            else:
                motif_start = int(chunk_range)
                motif_end = motif_start + 1

            motif_mask[:, curr_pos : curr_pos + motif_end - motif_start] = 1

            trans_locations.append((curr_pos, curr_pos + motif_end - motif_start))
            for atom, atom_locations in motif_locations.items():
                atom_locations.append(
                    motif_backbones[chunk_chain][atom][
                        motif_start - OFFSET : motif_end - OFFSET
                    ]
                )

            sum_of_coords += torch.sum(motif_locations[ATOM_TO_CENTER][-1], dim=0)
            n_motif_residues += motif_end - motif_start
            curr_pos += motif_end - motif_start
        else:
            if DASH in chunk_range:
                min_segment_length, max_segment_length = map(
                    int, chunk_range.split(DASH)
                )
                # Use median scaffold length similar to TDS paper
                scaffold_segment_length = (min_segment_length + max_segment_length) // 2
            else:
                scaffold_segment_length = int(chunk_range)

            curr_pos += scaffold_segment_length

        assert (
            curr_pos + 1 <= max_n_residues
        ), f"Exceeded experiment's maximum number of residues: ({curr_pos + 1} > {max_n_residues})"

    mask[:, : curr_pos + 1] = 1
    if not return_masked_backbones:
        return motif_mask, mask

    motif_mean = sum_of_coords / n_motif_residues

    out = {atom: torch.zeros((1, max_n_residues, 3)) for atom in ATOMS}
    for i, (start, end) in enumerate(trans_locations):
        for atom in ATOMS:
            out[atom][:, start:end] = motif_locations[atom][i] - motif_mean

    return motif_mask, mask, out


def split_multi_motif_spec(contig: str) -> Dict[str, str]:
    DASH = "-"

    common = []
    per_group_specs = {}

    is_range = re.compile(r"^([0-9]+)|([0-9]+\-[0-9]+)$")
    motif_segment = re.compile(
        r"^(?P<motif>[a-zA-Z0-9]+)"
        r"/(?P<chain>[a-zA-Z]?)(?P<range>([0-9]+)|([0-9]+\-[0-9]+))"
        r"\{(?P<group_no>[0-9]+)\}$"
    )
    for chunk in contig.split(","):
        if is_range.fullmatch(chunk):
            common.append(chunk)
            for group_specs in per_group_specs.values():
                for motif_specs in group_specs.values():
                    motif_specs.append(chunk)
            continue

        motif_chunk = motif_segment.fullmatch(chunk).groupdict()
        motif = motif_chunk["motif"]
        chunk_chain = motif_chunk["chain"]
        chunk_range = motif_chunk["range"]
        group_no = int(motif_chunk["group_no"])

        if group_no not in per_group_specs:
            per_group_specs[group_no] = {}
        if motif not in per_group_specs[group_no]:
            per_group_specs[group_no][motif] = [] + common

        if DASH in chunk_range:
            start, end = map(int, chunk_range.split(DASH))
            chunk_length = str(end - start + 1)
        else:
            chunk_length = str(1)
        common.append(chunk_length)
        for group_specs in per_group_specs.values():
            for motif_specs in group_specs.values():
                motif_specs.append(chunk_length)

        per_group_specs[group_no][motif][-1] = chunk_chain + chunk_range

    for group_specs in per_group_specs.values():
        for motif, motif_specs in group_specs.items():
            group_specs[motif] = ",".join(motif_specs)

    return per_group_specs
