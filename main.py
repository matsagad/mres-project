from utils.path import add_submodules_to_path

add_submodules_to_path()

from experiments import EXPERIMENTS_REGISTRY, placeholder_job
import hydra
import logging
from omegaconf import DictConfig, OmegaConf
import os
import sys
import traceback
from types import TracebackType

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="config", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)

    experiment_name = cfg.experiment["name"]
    EXPERIMENTS_REGISTRY.get(experiment_name, placeholder_job)(cfg)


def log_exception(
    exc_type: type, exc_value: BaseException, exc_traceback: TracebackType
) -> None:
    exc_info = (exc_type, exc_value, exc_traceback)
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(*exc_info)
        return
    logger.error("".join(traceback.format_exception(*exc_info)))


# Need to set environment variable to enable override of sys.excepthook
os.environ["HYDRA_FULL_ERROR"] = "1"
sys.excepthook = log_exception

if __name__ == "__main__":
    main()
