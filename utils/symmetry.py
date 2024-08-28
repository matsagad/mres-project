def get_n_symmetries(symmetry: str):
    if symmetry.startswith("C-"):
        return int(symmetry.split("-")[-1])
    if symmetry.startswith("D-"):
        return 2 * int(symmetry.split("-")[-1])
    if symmetry == "T":
        return 12
    if symmetry == "O":
        return 24
    raise Exception(f"Unsupported symmetry group {symmetry}.")
