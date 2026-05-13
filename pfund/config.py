# pyright: reportUnusedParameter=false
from __future__ import annotations
from typing import Any

from pathlib import Path

from pfund.enums import Environment
from pfund_kit import logging as kit_logging
from pfund_kit.config import Configuration


__all__ = [
    'get_config',
    'configure',
    'configure_logging',
    'setup_logging',
]


project_name = 'pfund'
_config: PFundConfig | None = None


def setup_logging(env: Environment, reset: bool = False) -> None:
    env = Environment[env.upper()]
    kit_logging.setup_logging(get_config(), env=env, reset=reset)


def get_config(engine_name: str | None = None) -> PFundConfig:
    """Lazy singleton - only creates config when first called.
    Also loads the .env file.
    """
    global _config
    if _config is None:
        _config = PFundConfig(engine_name=engine_name)
    if engine_name is not None and _config.engine_name is not None:
        config_engine_name = _config.engine_name
        assert engine_name == config_engine_name, f'global config is already set using engine name {config_engine_name}'
    return _config


def get_logging_config() -> dict[str, Any]:
    return kit_logging.get_logging_config(get_config())


def configure(
    engine_name: str | None = None,
    data_path: str | None = None,
    log_path: str | None = None,
    cache_path: str | None = None,
    persist: bool = False,
) -> PFundConfig:
    '''
    Configures the global config object.
    Args:
        engine_name: if engine name is provided, the config per engine will be set.
            if not provided, the global config will be set.
        data_path: Path to the data directory.
        log_path: Path to the log directory.
        cache_path: Path to the cache directory.
        persist: If True, the config will be saved to the config file.
    '''
    config = get_config(engine_name=engine_name)
    config_dict = config.to_dict()
    config_dict.pop('__version__')

    # Apply updates for non-None values
    for k in config_dict:
        v = locals().get(k)
        if v is not None:
            if '_path' in k:
                v = Path(v)
            setattr(config, k, v)

    config.ensure_dirs()

    if persist:
        config.save()

    return config


def configure_logging(logging_config: dict[str, Any] | None = None, debug: bool = False) -> dict[str, Any]:
    return kit_logging.configure_logging(get_config(), overrides=logging_config, debug=debug)


class PFundConfig(Configuration):
    SETTINGS_FILENAME = 'settings.toml'  # engine's settings toml file

    def __init__(self, engine_name: str | None = None):
        self._engine_name = engine_name
        super().__init__(project_name=project_name, source_file=__file__)
        self.settings_file_path.parent.mkdir(parents=True, exist_ok=True)
        self.settings_file_path.touch(exist_ok=True)

    def _initialize_from_data(self):
        """Initialize PFundConfig-specific attributes from config data."""
        pass

    @property
    def engine_name(self) -> str | None:
        return self._engine_name

    @property
    def log_path(self) -> Path:
        base = self._log_path
        return base / self._engine_name if self._engine_name else base
    
    @log_path.setter
    def log_path(self, value: Path):
        self._log_path = Path(value)

    @property
    def data_path(self) -> Path:
        base = self._data_path
        return base / self._engine_name if self._engine_name else base
    
    @data_path.setter
    def data_path(self, value: Path):
        self._data_path = Path(value)

    @property
    def cache_path(self) -> Path:
        base = self._cache_path
        return base / self._engine_name if self._engine_name else base

    @cache_path.setter
    def cache_path(self, value: Path):
        self._cache_path = Path(value)

    @property
    def settings_file_path(self) -> Path:
        if self._engine_name:
            return self.config_path / self._engine_name / self.SETTINGS_FILENAME
        return self.config_path / self.SETTINGS_FILENAME

    def prepare_docker_context(self):
        pass
    