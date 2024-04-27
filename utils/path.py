import hydra
import sys

def add_submodules_to_path():
    sys.path.append("submodules/genie")

def out_dir() -> str:
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    return hydra_cfg["runtime"]["output_dir"]