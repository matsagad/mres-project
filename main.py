from utils.path import add_submodules_to_path

add_submodules_to_path()

from experiments import EXPERIMENTS_REGISTRY, placeholder_job
import hydra
import logging
from omegaconf import DictConfig, OmegaConf
import sys
import traceback
from types import TracebackType

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="config", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)

    experiment_name = cfg.experiment["name"]
    EXPERIMENTS_REGISTRY.get(experiment_name, placeholder_job)(cfg)


def log_exception(_type: type, value: BaseException, tb: TracebackType) -> None:
    if issubclass(_type, KeyboardInterrupt):
        sys.__excepthook__(_type, value, tb)
        return
    logger.exception("".join(traceback.format_exception(_type, value, tb)))


sys.excepthook = log_exception

if __name__ == "__main__":
    main()
