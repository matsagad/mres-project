from conditional.wrapper import ConditionalWrapper, ConditionalWrapperConfig
import pathlib
from typing import Callable, Dict, Tuple
from utils.path import import_all_files_in_directory

CONDITIONAL_METHOD_REGISTRY: Dict[
    str, Tuple[ConditionalWrapper, Callable[[Dict[str, any]], Dict[str, any]]]
] = {}


def register_conditional_method(
    name: str, config: ConditionalWrapperConfig
) -> Callable[[ConditionalWrapper], ConditionalWrapper]:

    def register(conditional_wrapper: ConditionalWrapper) -> ConditionalWrapper:
        if name in CONDITIONAL_METHOD_REGISTRY:
            raise Exception(f"Conditional method '{name}' already registered!")

        config_resolver = config.get_resolver(name)
        CONDITIONAL_METHOD_REGISTRY[name] = (conditional_wrapper, config_resolver)

        return conditional_wrapper

    return register


# Load all conditional methods to populate registry
import_all_files_in_directory(pathlib.Path(__file__).parent.resolve(), "conditional")
