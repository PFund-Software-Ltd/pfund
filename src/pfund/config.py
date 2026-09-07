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
    env: Environment | str,
    reset: bool = False,
    config: PFundConfig | None = None,
) -> None:
    """Args:
    config: the engine's config, whose paths are scoped to its name. Defaults
        to the global config, which is not scoped to any engine.
    """
    env = Environment[env.upper()]
    kit_logging.setup_logging(config or get_config(), env=env, reset=reset)


def get_config() -> PFundConfig:
    global _config
    if _config is None:
        _config = PFundConfig()
    return _config


def get_logging_config() -> dict[str, Any]:
    return kit_logging.get_logging_config(get_config())


def configure(
    data_path: str | None = None,
    log_path: str | None = None,
    cache_path: str | None = None,
    show_progress_bar: bool | None = None,
    persist: bool = False,
) -> PFundConfig:
    """
    Configures the global config object.
    Args:
        data_path: Path to the data directory.
        log_path: Path to the log directory.
        cache_path: Path to the cache directory.
        show_progress_bar: Whether pfund progress bars are displayed.
        persist: If True, the config will be saved to the config file.
    """
    config = get_config()
    config_dict = config.to_dict()

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
        self.show_progress_bar = self._data.get("show_progress_bar", True)

    def get_settings_file_path(self, engine_name: str) -> Path:
        """Where an engine's settings.toml lives, one directory per engine.

        Here rather than on the engine context so that the CLI, which has no
        engine, resolves the same path from the same place.
        """
        return self.config_path / engine_name / self.SETTINGS_FILENAME

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "show_progress_bar": self.show_progress_bar,
        }

    def prepare_docker_context(self):
        pass
