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

        # Resolve correct types for config of conditional wrapper
        def config_resolver(config_values: Dict[str, any]) -> Dict[str, any]:
            wrapper_config = {}
            for key, value in config_values.items():
                if key not in config.__annotations__:
                    assert (
                        key == "name" or key == "method"
                    ), f"Did you forget to add '{key}' key in the {name} config class and/or YAML file?"
                    continue
                try:
                    wrapper_config[key] = (
                        None if value is None else config.__annotations__[key](value)
                    )
                except:
                    raise Exception(
                        f"Expected type {config.__annotations__[key]} for parameter {key} but got {value}."
                    )
            return wrapper_config

        CONDITIONAL_METHOD_REGISTRY[name] = (conditional_wrapper, config_resolver)

        return conditional_wrapper

    return register


# Load all conditional methods to populate registry
import_all_files_in_directory(pathlib.Path(__file__).parent.resolve(), "conditional")
