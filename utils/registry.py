from typing import Callable, Dict


class ConfigOutline:
    """
    Base class for configs. Used throughout to resolve typing within config values.
    """

    @classmethod
    def get_resolver(cls, name: str) -> Callable[[Dict[str, any]], Dict[str, any]]:

        # Resolve correct types for given config values.
        def config_resolver(config_values: Dict[str, any]) -> Dict[str, any]:
            # We choose to only pick out parameters listed in the config outline
            # and therefore silently ignore additional parameters in the given
            # config_values.
            wrapper_config = {}
            for param in cls.__annotations__.keys():
                if param not in config_values:
                    raise Exception(
                        f"Did you forget to add '{param}' in the {name} YAML file? "
                    )
                value = config_values[param]
                try:
                    wrapper_config[param] = (
                        None if value is None else cls.__annotations__[param](value)
                    )
                except:
                    raise Exception(
                        f"Expected type {cls.__annotations__[param]} for parameter {param} but got {value}."
                    )
            return wrapper_config

        return config_resolver
