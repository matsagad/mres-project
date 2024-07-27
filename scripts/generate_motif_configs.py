import argparse
import os
import pandas as pd
import re
from typing import Dict


def main(args: Dict[str, any]):
    NEWLINE = "\n"
    df = pd.read_csv(args.motif_benchmark_csv, index_col=0)

    for _, row in df.iterrows():
        yaml_data = {
            "name": row["target"],
            "path": row["motif_path"],
            "contig": row["contig"],
        }
        with open(
            os.path.join(
                args.config_dir, "experiment", "motif", f"{row['target']}.yaml"
            ),
            "w",
        ) as f_out:
            f_out.write(NEWLINE.join(f"{k}: {v}" for k, v in yaml_data.items()))

    df = pd.read_csv(args.multi_motif_benchmark_csv, index_col=0)
    motif_path_exp = re.compile("(?P<dir_name>.*/?)\{(?P<motifs>[a-zA-Z0-9,]+)\}.pdb")
    for _, row in df.iterrows():
        path_match = motif_path_exp.match(row["motif_paths"])
        dir_name = path_match.group("dir_name")
        motifs = path_match.group("motifs").split(",")

        yaml_data = {
            "name": row["target"],
            "paths": [os.path.join(dir_name, f"{motif}.pdb") for motif in motifs],
            "contig": row["contig"],
        }
        with open(
            os.path.join(
                args.config_dir, "experiment", "multi_motif", f"{row['target']}.yaml"
            ),
            "w",
        ) as f_out:
            f_out.write(NEWLINE.join(f"{k}: {v}" for k, v in yaml_data.items()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--motif_benchmark_csv", default="data/motif_problems/benchmark.csv"
    )
    parser.add_argument(
        "--multi_motif_benchmark_csv", default="data/multi_motif_problems/benchmark.csv"
    )
    parser.add_argument("--config_dir", default="config")
    args = parser.parse_args()

    main(args)
