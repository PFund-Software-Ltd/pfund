# pyright: reportUnusedParameter=false
from __future__ import annotations
from typing import Literal, Any

from pathlib import Path

from pfeed.enums import IOFormat, DataStorage
from pfund.enums import Environment, Database
from pfund_kit.config import Configuration


__all__ = [
    'get_config',
    'configure',
    'configure_logging',
    'setup_logging',
]


project_name = 'pfund'
_config: PFundConfig | None = None
_logging_config: dict[str, Any] | None = None


def setup_logging(env: Environment, engine_name: str, reset: bool=False):
    from pfund_kit.logging import clear_logging_handlers, setup_exception_logging
    from pfund_kit.logging.configurator import LoggingDictConfigurator
    
    env = Environment[env.upper()]
    
    if reset:
        clear_logging_handlers()
        
    config: PFundConfig = get_config()
    logging_config: dict[str, Any] = get_logging_config()

    log_path = config.log_path / env / engine_name
    log_path.mkdir(parents=True, exist_ok=True)

    # ≈ logging.config.dictConfig(logging_config) with a custom configurator
    logging_configurator: LoggingDictConfigurator = LoggingDictConfigurator.create(
        log_path=log_path, 
        logging_config=logging_config, 
        lazy=True,
        use_colored_logger=True,
    )
    logging_configurator.configure()
    
    setup_exception_logging(logger_name=project_name)


def get_config() -> PFundConfig:
    """Lazy singleton - only creates config when first called.
    Also loads the .env file.
    """
    global _config
    if _config is None:
        _config = PFundConfig()
    return _config


def get_logging_config() -> dict[str, Any]:
    global _logging_config
    if _logging_config is None:
        _logging_config = configure_logging()
    return _logging_config


def configure(
    data_path: str | None = None,
    log_path: str | None = None,
    cache_path: str | None = None,
    database: Literal[Database.DUCKDB, 'duckdb'] | None = None,
    storage: DataStorage | str | None = None,
    storage_options: dict[DataStorage | str, dict[str, Any]] | None = None,
    io_options: dict[IOFormat | str, dict[str, Any]] | None = None,
    persist: bool = False,
) -> PFundConfig:
    '''
    Configures the global config object.
    Args:
        data_path: Path to the data directory.
        log_path: Path to the log directory.
        cache_path: Path to the cache directory.
        database: database for storing engine states, e.g. "duckdb"
        storage: storage for storing component data, e.g. "local"
            if database is file-based (e.g. .duckdb), storage will also store it
        storage_options: storage options for data storages, used by database and storage
        io_options: io options for io formats, used by database and storage
        persist: If True, the config will be saved to the config file.
    '''
    config = get_config()
    config_dict = config.to_dict()
    config_dict.pop('__version__')
    
    # Apply updates for non-None values
    for k in config_dict:
        v = locals().get(k)
        if v is not None:
            if '_path' in k:
                v = Path(v)
            elif k == 'database':
                v = Database[v.upper()]
            elif k == 'storage':
                v = DataStorage[v.upper()]
            elif k == 'storage_options':
                v = { DataStorage[k.upper()]: v for k, v in v.items() }
            elif k == 'io_options':
                v = { IOFormat[k.upper()]: v for k, v in v.items() }
            setattr(config, k, v)
    
    config.ensure_dirs()
    
    if persist:
        config.save()
        
    return config


def configure_logging(logging_config: dict[str, Any] | None=None, debug: bool=False) -> dict[str, Any]:
    '''
    Loads logging config from YAML file and merges with optional user overrides.

    Args:
        logging_config: Optional dict to override/extend the base YAML config.
        debug: If True, sets all loggers and handlers to DEBUG level.
               This overrides any level settings from the YAML file and logging_config.

    Returns:
        Merged logging config dict.

    Raises:
        FileNotFoundError: If the logging config YAML file is not found.
    '''
    from pfund_kit.utils import deep_merge
    from pfund_kit.logging import enable_debug_logging
    from pfund_kit.utils.yaml import load

    global _logging_config

    config = get_config()

    # load logging.yml file
    logging_config_from_yml: dict | None = load(config.logging_config_file_path)
    if logging_config_from_yml is None:
        raise FileNotFoundError(f"Logging config file {config.logging_config_file_path} not found")
    
    _logging_config = deep_merge(logging_config_from_yml, logging_config or {})
    if debug:
        _logging_config = enable_debug_logging(_logging_config)
    return _logging_config
    

class PFundConfig(Configuration):
    SETTINGS_FILENAME = 'settings.toml'  # engine's settings toml file
    
    def __init__(self):
        super().__init__(project_name=project_name, source_file=__file__)
        self.settings_file_path.touch(exist_ok=True)

    def _initialize_from_data(self):
        """Initialize PFundConfig-specific attributes from config data."""
        self.database = Database[self._data.get('database', Database.DUCKDB).upper()]
        self.storage = DataStorage[self._data.get('storage', DataStorage.LOCAL).upper()]
        self.storage_options = self._data.get('storage_options', {})
        self.io_options = self._data.get('io_options', {})
    
    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            'database': self.database,
            'storage': self.storage,
            'storage_options': self.storage_options,
            'io_options': self.io_options,
        }

    @property
    def settings_file_path(self):
        return self.config_path / self.SETTINGS_FILENAME
    
    def prepare_docker_context(self):
        pass
    