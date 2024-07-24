from model.diffusion import DiffusionModelConfig, FrameDiffusionModel
import pathlib
from typing import Callable, Dict, Tuple
from utils.path import import_all_files_in_directory

DIFFUSION_MODEL_REGISTRY: Dict[
    str, Tuple[FrameDiffusionModel, Callable[[Dict[str, any]], Dict[str, any]]]
] = {}


def register_diffusion_model(
    name: str, config: DiffusionModelConfig
) -> Callable[[FrameDiffusionModel], FrameDiffusionModel]:

    def register(model: FrameDiffusionModel) -> FrameDiffusionModel:
        if name in DIFFUSION_MODEL_REGISTRY:
            raise Exception(f"Diffusion model '{name}' already registered!")

        config_resolver = config.get_resolver(name)
        DIFFUSION_MODEL_REGISTRY[name] = (model, config_resolver)

        return model

    return register


# Load all conditional methods to populate registry
import_all_files_in_directory(pathlib.Path(__file__).parent.resolve(), "model")
