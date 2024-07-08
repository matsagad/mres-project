import argparse
import os
import pandas as pd
from typing import Dict


def main(args: Dict[str, any]):
    NEWLINE = "\n"
    df = pd.read_csv(args.benchmark_csv, index_col=0)

    for _, row in df.iterrows():
        yaml_data = {
            "name": row["target"],
            "path": row["motif_path"],
            "contig_region": row["contig"],
        }
        with open(os.path.join(args.config_dir, f"{row['target']}.yaml"), "w") as f_out:
            f_out.write(NEWLINE)
            f_out.write(NEWLINE.join(f"{k}: {v}" for k, v in yaml_data.items()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_csv", default="data/motif_problems/benchmark.csv")
    parser.add_argument("--config_dir", default="config/experiment/motif")
    args = parser.parse_args()

    main(args)
