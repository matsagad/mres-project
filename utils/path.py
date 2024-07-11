import hydra
import importlib
import os
import pathlib
import re
import sys


def add_submodules_to_path():
    sys.path.append("submodules/genie")


def out_dir() -> str:
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    return hydra_cfg["runtime"]["output_dir"]


def import_all_files_in_directory(path: str, curr_package: str) -> None:
    EXPOSED_PYTHON_FILE = re.compile("^(?![_]).*\.py$")
    with os.scandir(path) as files:
        for file in files:
            if not file.is_file() or EXPOSED_PYTHON_FILE.match(file.name) is None:
                continue
            importlib.import_module("." + pathlib.Path(file.name).stem, curr_package)
