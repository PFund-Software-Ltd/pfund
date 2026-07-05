# pyright: reportUnusedParameter=false
from __future__ import annotations

from pathlib import Path
from typing import Any

from pfund_kit import logging as kit_logging
from pfund_kit.config import Configuration

from pfund.enums import Environment

__all__ = [
    "configure",
    "configure_logging",
    "get_config",
    "setup_logging",
]


project_name = "pfund"
_config: PFundConfig | None = None


def setup_logging(
    env: Environment | str, engine_name: str, reset: bool = False
) -> None:
    env = Environment[env.upper()]
    config_by_engine_name = get_config(engine_name=engine_name)
    kit_logging.setup_logging(config_by_engine_name, env=env, reset=reset)


def get_config(engine_name: str | None = None) -> PFundConfig:
    """Lazy singleton - only creates config when first called.
    Also loads the .env file.
    """
    global _config
    if _config is None:
        _config = PFundConfig()
    if engine_name is None:
        return _config
    else:
        # modify the config based on the engine name on the fly so that the global config is intact
        from copy import copy

        config_by_engine_name = copy(_config)
        config_by_engine_name.log_path /= engine_name
        config_by_engine_name.data_path /= engine_name
        config_by_engine_name.cache_path /= engine_name
        return config_by_engine_name


def get_logging_config() -> dict[str, Any]:
    return kit_logging.get_logging_config(get_config())


def configure(
    data_path: str | None = None,
    log_path: str | None = None,
    cache_path: str | None = None,
    persist: bool = False,
) -> PFundConfig:
    """
    Configures the global config object.
    Args:
        data_path: Path to the data directory.
        log_path: Path to the log directory.
        cache_path: Path to the cache directory.
        persist: If True, the config will be saved to the config file.
    """
    config = get_config()
    config_dict = config.to_dict()
    config_dict.pop("__version__")

    # Apply updates for non-None values
    for k in config_dict:
        v = locals().get(k)
        if v is not None:
            if "_path" in k:
                v = Path(v)
            setattr(config, k, v)

    config.ensure_dirs()

    if persist:
        config.save()

    return config


def configure_logging(
    logging_config: dict[str, Any] | None = None, debug: bool = False
) -> dict[str, Any]:
    return kit_logging.configure_logging(
        get_config(), overrides=logging_config, debug=debug
    )


class PFundConfig(Configuration):
    SETTINGS_FILENAME = "settings.toml"  # engine's settings toml file

    def __init__(self):
        super().__init__(project_name=project_name, source_file=__file__)

    def _initialize_from_data(self):
        """Initialize PFundConfig-specific attributes from config data."""
        pass

    def get_settings_file_path(self, engine_name: str) -> Path:
        settings_file_path = self.config_path / engine_name / self.SETTINGS_FILENAME
        settings_file_path.parent.mkdir(parents=True, exist_ok=True)
        settings_file_path.touch(exist_ok=True)
        return settings_file_path

    def prepare_docker_context(self):
        pass
