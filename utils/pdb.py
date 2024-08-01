import re
from torch import Tensor
import torch
from typing import Dict, List, Tuple, Union


def pdb_to_atom_backbone(
    f_pdb: str, atoms: List[str] = ["N", "CA", "C", "O"]
) -> Dict[str, Dict[str, Tensor]]:
    """
    Given a PDB file, it retrieves the 3D coordinates of specified
    backbone atoms in each chain of the protein. I.e. the output is a
    dictionary that maps each chain to a dictionary of each atom and
    its 3D coordinates.
    """
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


def pdb_to_c_alpha_backbone(f_pdb: str, chain: str = "A") -> Tensor:
    """
    Given a PDB file, extract the C-alpha backbone coordinates of
    a specified chain.
    """
    return pdb_to_atom_backbone(f_pdb)[chain]["CA"]


def atom_backbone_to_pdb(
    backbones: Dict[str, Tensor],
    f_out: str,
    placeholder_aa: str = "ALA",
    placeholder_chain: str = "A",
    atoms: List[str] = ["N", "CA", "C", "O"],
) -> None:
    """
    Converts a given backbone dictionary (mapping atom to coordinates)
    into a PDB file. Note that atom indices start at one.
    """
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
    atom_i = 1
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
                    res_idx=i + 1,
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
    """
    Convert 3D coordinates of the given C-alpha backbone into a PDB file.
    """
    atom_backbone_to_pdb(
        {"CA": backbone.squeeze()},
        f_out,
        placeholder_aa,
        placeholder_chain,
        atoms=["CA"],
    )


def create_evaluation_motif_pdb(
    f_out: str,
    f_pdb: str,
    motif_mask: Tensor,
    contig: str,
) -> None:
    """
    Filter existing motif PDB file so that:
        - the chain indices are modified to match the motif mask,
        - and only coordinates specified in the spec/contig remain.
    """
    eval_motif_pdb_str = _get_evaluation_motif_pdb_str(f_pdb, motif_mask, contig, "A")

    with open(f_out, "w") as f:
        f.write(eval_motif_pdb_str)


def create_evaluation_multi_motif_pdb(
    f_out: str,
    f_pdbs: Dict[str, str],
    unmerged_motif_masks: Dict[int, Dict[str, Tensor]],
    per_group_specs: Dict[int, Dict[str, str]],
) -> None:
    """
    Filter existing motif PDB files and stack them into one so that:
        - the chain indices are modified to match the motif mask,
        - only coordinates specified in the spec/contig remain,
        - motifs in different groups are labelled in the segment id column,
        - and the output file is stacked in order of chain index, regardless
          of which group and protein it belongs to.
    """
    GROUP_NO_OFFSET = 1
    segment_ids = [chr(ord("A") + group_no) for group_no in range(len(per_group_specs))]

    overall_segmented_pdb = {}
    for group_no, group_specs in per_group_specs.items():
        for motif_name, motif_specs in group_specs.items():
            segmented_pdb = _get_segmented_motif_pdb_by_index(
                f_pdbs[motif_name],
                unmerged_motif_masks[group_no][motif_name],
                motif_specs,
                segment_ids[group_no - GROUP_NO_OFFSET],
            )

            assert not set(segmented_pdb.keys()).intersection(
                set(overall_segmented_pdb.keys())
            )
            overall_segmented_pdb.update(segmented_pdb)

    filtered_output = []
    for chain_index in sorted(overall_segmented_pdb.keys()):
        filtered_output.append(overall_segmented_pdb[chain_index])

    with open(f_out, "w") as f:
        f.write("".join(filtered_output))


def _get_evaluation_motif_pdb_str(
    f_pdb: str,
    motif_mask: Tensor,
    contig: str,
    segment_id: str,
) -> str:
    segmented_pdb = _get_segmented_motif_pdb_by_index(
        f_pdb, motif_mask, contig, segment_id
    )
    filtered_output = []
    for chain_index in sorted(segmented_pdb.keys()):
        filtered_output.append(segmented_pdb[chain_index])
    return "".join(filtered_output)


def _get_segmented_motif_pdb_by_index(
    f_pdb: str,
    motif_mask: Tensor,
    contig: str,
    segment_id: str,
) -> Dict[int, str]:
    COMMA = ","
    DASH = "-"
    NEWLINE = "\n"
    PDB_INDEX_OFFSET = 1
    chain_segment = re.compile(
        r"(?P<chain>[a-zA-Z]?)(?P<range>([0-9]+)|([0-9]+\-[0-9]+))"
    )

    segments_to_keep = {}

    motif_index = 0
    motif_indices = (
        torch.argwhere(motif_mask[0].cpu()).squeeze() + PDB_INDEX_OFFSET
    ).tolist()

    for chunk in contig.split(COMMA):
        chunk_dict = chain_segment.fullmatch(chunk).groupdict()
        chunk_chain = chunk_dict["chain"]
        chunk_range = chunk_dict["range"]
        if not chunk_chain:
            continue
        if chunk_chain not in segments_to_keep:
            segments_to_keep[chunk_chain] = {}
        if DASH in chunk_range:
            start, end = map(int, chunk_range.split(DASH))
            for i in range(start, end + 1):
                segments_to_keep[chunk_chain][i] = motif_indices[motif_index]
                motif_index += 1
        else:
            index = int(chunk_range)
            segments_to_keep[chunk_chain][index] = motif_indices[motif_index]
            motif_index += 1

    LABELS = "xyzot"
    atom_line = re.compile(
        rf"^ATOM\s*[0-9]+\s*(?P<atom>[a-zA-Z]*)"
        + r"\s+[a-zA-Z]+\s+(?P<chain>[a-zA-Z])(?P<chain_index>\s*[0-9]+)\s*"
        + r"\s*".join(rf"(?P<{label}>[0-9\.\-]+)" for label in LABELS)
        + r"\s+(?P<segment_id>[a-zA-Z]?)\s+(?P<element>[a-zA-Z0-9\+\-])"
    )

    lines_by_chain_index = {}
    with open(f_pdb, "r") as f:
        for line in map(str.rstrip, f.readlines()):
            match = atom_line.match(line)
            if not match:
                continue

            match_dict = match.groupdict()
            match_chain = match_dict["chain"]
            match_chain_index = match_dict["chain_index"]
            if (
                match_chain in segments_to_keep
                and int(match_chain_index) in segments_to_keep[match_chain]
            ):
                chain_start = match.start("chain_index")
                chain_end = match.end("chain_index")
                chain_index_str_len = chain_end - chain_start

                segment_id_start = match.start("segment_id")
                segment_id_end = match.end("segment_id")
                segment_id_str_len = segment_id_end - segment_id_start

                new_chain_index = segments_to_keep[match_chain][int(match_chain_index)]
                if new_chain_index not in lines_by_chain_index:
                    lines_by_chain_index[new_chain_index] = []

                lines_by_chain_index[new_chain_index].extend(
                    (
                        line[:chain_start],
                        str(
                            segments_to_keep[match_chain][int(match_chain_index)]
                        ).rjust(chain_index_str_len),
                        line[chain_end:segment_id_start],
                        segment_id.rjust(segment_id_str_len),
                        line[segment_id_end:],
                        NEWLINE,
                    )
                )

        segmented_pdb = {
            chain_index: "".join(chain_index_lines)
            for chain_index, chain_index_lines in lines_by_chain_index.items()
        }

    return segmented_pdb


def get_motif_mask(
    motif_backbones: Dict[str, Dict[str, Tensor]],
    max_n_residues: int,
    contig: str = None,
    return_masked_backbones: bool = False,
) -> Union[Tuple[Tensor, Tensor, Dict[str, Tensor]], Tuple[Tensor, Tensor]]:
    """
    Get motif coordinates and mask given the backbone of the parent protein
    and the spec/contig specifying which coordinates to include/design.
    """
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


def split_multi_motif_spec(contig: str) -> Dict[str, Dict[str, str]]:
    """
    Given a spec/contig for a multi-motif problem (see Genie2 paper),
    break it down to several single motif specs, grouped by their
    motif group and their parent protein name.
    """
    COMMA = ","
    DASH = "-"

    common = []
    per_group_specs = {}

    is_range = re.compile(r"^([0-9]+)|([0-9]+\-[0-9]+)$")
    motif_segment = re.compile(
        r"^(?P<motif>[a-zA-Z0-9]+)"
        r"/(?P<chain>[a-zA-Z]?)(?P<range>([0-9]+)|([0-9]+\-[0-9]+))"
        r"\{(?P<group_no>[0-9]+)\}$"
    )
    for chunk in contig.split(COMMA):
        if is_range.fullmatch(chunk):
            common.append(chunk)
            for group_specs in per_group_specs.values():
                for motif_specs in group_specs.values():
                    motif_specs.append(chunk)
            continue

        motif_chunk = motif_segment.fullmatch(chunk).groupdict()
        motif_name = motif_chunk["motif"]
        chunk_chain = motif_chunk["chain"]
        chunk_range = motif_chunk["range"]
        group_no = int(motif_chunk["group_no"])

        if group_no not in per_group_specs:
            per_group_specs[group_no] = {}
        if motif_name not in per_group_specs[group_no]:
            per_group_specs[group_no][motif_name] = [] + common

        if DASH in chunk_range:
            start, end = map(int, chunk_range.split(DASH))
            chunk_length = str(end - start + 1)
        else:
            chunk_length = str(1)
        common.append(chunk_length)
        for group_specs in per_group_specs.values():
            for motif_specs in group_specs.values():
                motif_specs.append(chunk_length)

        per_group_specs[group_no][motif_name][-1] = chunk_chain + chunk_range

    for group_specs in per_group_specs.values():
        for motif_name, motif_specs in group_specs.items():
            group_specs[motif_name] = ",".join(motif_specs)

    return per_group_specs


def find_all_masks_satisfying_spec(
    contig: str, length_range: Tuple[int, int]
) -> Tuple[Tensor, Tensor]:
    """
    Find all possible motif masks defined by a spec/contig that
    satisfy the length range specified.
    """
    COMMA = ","
    DASH = "-"
    chain_segment = re.compile(
        r"(?P<chain>[a-zA-Z]?)(?P<range>([0-9]+)|([0-9]+\-[0-9]+))"
    )

    ranges = []
    is_motif = []
    for chunk in contig.split(COMMA):
        chunk_dict = chain_segment.fullmatch(chunk).groupdict()
        chunk_chain = chunk_dict["chain"]
        chunk_range = chunk_dict["range"]

        is_motif.append(1 if chunk_chain else 0)

        if DASH in chunk_range:
            index_range = map(int, chunk_range.split(DASH))
            if chunk_chain:
                start, end = index_range
                length = end - start + 1
                ranges.append((length, length))
            else:
                min_length, max_length = index_range
                ranges.append((min_length, max_length))
        else:
            if chunk_chain:
                ranges.append((1, 1))
            else:
                length = int(chunk_range)
                ranges.append((length, length))

    solutions = []
    min_target, max_target = length_range
    _find_solutions([], solutions, ranges[::-1], 0, min_target, max_target)

    solutions = torch.tensor(solutions)
    is_motif = torch.tensor(is_motif)

    return solutions, is_motif


def _find_solutions(
    curr: List[int],
    solutions: List[List[int]],
    backlog: List[Tuple[int, int]],
    acc: int,
    min_target: int,
    max_target: int,
):
    if not backlog:
        if min_target <= acc and acc <= max_target:
            solutions.append(curr)
        return
    min_length, max_length = backlog.pop()
    for length in range(min_length, max_length + 1):
        if acc + length > max_target:
            continue
        _find_solutions(
            curr + [length], solutions, backlog, acc + length, min_target, max_target
        )
    backlog.append((min_length, max_length))


def get_motifs_and_masks_for_all_placements(
    mask: Tensor, motif: Tensor, motif_mask: Tensor, contig: str
) -> Tuple[Tensor, Tensor]:
    """
    In the case where placement of motif is variable, get all possible
    motif and motif masks that can be made to fit the spec/contig.
    Note: we use the length of the protein mask as both the min and max
    possible lengths. As shown in 'get_motif_mask' above, we find the
    median of each range to determine the overall length.
    """
    N_RESIDUES = (mask[0] == 1).sum()
    placements, motif_placement_mask = find_all_masks_satisfying_spec(
        contig, (N_RESIDUES, N_RESIDUES)
    )

    motifs = []
    motif_masks = []
    for placement in placements:
        motif_i = torch.zeros(motif.shape, device=motif.device)
        motif_mask_i = torch.zeros(motif_mask.shape, device=motif_mask.device)

        motif_range = torch.repeat_interleave(motif_placement_mask, placement) == 1
        motif_i[:, :N_RESIDUES][:, motif_range] = motif[:, motif_mask[0] == 1]
        motif_mask_i[:, :N_RESIDUES][:, motif_range] = 1

        motifs.append(motif_i)
        motif_masks.append(motif_mask_i)
    motifs = torch.cat(motifs)
    motif_masks = torch.cat(motif_masks)

    return motifs, motif_masks
