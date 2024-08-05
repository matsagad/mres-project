import gc
import logging
from omegaconf import DictConfig
import pathlib
import time
import torch
from typing import Callable, Dict
from utils.path import import_all_files_in_directory, out_dir

EXPERIMENTS_REGISTRY: Dict[str, Callable[[DictConfig], None]] = {}

logger = logging.getLogger(__name__)


def register_experiment(name: str) -> Callable:
    def experiment_job(fn: Callable) -> Callable:
        def wrapper(cfg: DictConfig):
            experiment_name = cfg.experiment["name"]

            t_start = time.time()
            logger.info(f"Started {experiment_name} experiment.")

            fn(cfg)

            t_elapsed_str = time.strftime(
                f"%H:%M:%S", time.gmtime(time.time() - t_start)
            )
            logger.info(f"Finished {experiment_name} experiment in {t_elapsed_str}.")
            logger.info(f"Output folder can be found at: {out_dir()}")
            gc.collect()
            torch.cuda.empty_cache()

        if name in EXPERIMENTS_REGISTRY:
            raise Exception(f"Experiment '{name}' already registered!")
        EXPERIMENTS_REGISTRY[name] = wrapper

        return wrapper

    return experiment_job


NO_EXPERIMENT_SET = "none"


@register_experiment(NO_EXPERIMENT_SET)
def placeholder_job(*_) -> None:
    print("Run with the --help flag to see options for experiments.")


# Load all experiments to populate registry
import_all_files_in_directory(pathlib.Path(__file__).parent.resolve(), "experiments")
