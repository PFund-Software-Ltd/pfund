from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfeed.enums import DataStorage

from pathlib import Path

from pfund.enums import Environment
from pfund_kit.config import Configuration


__all__ = [
    'get_config',
    'configure',
    'configure_logging',
    'setup_logging',
]


project_name = 'pfund'
_config: PFundConfig | None = None
_logging_config: dict | None = None


def setup_logging(env: Environment | None=None, reset: bool=False):
    from pfund_kit.logging import clear_logging_handlers, setup_exception_logging
    from pfund_kit.logging.configurator import LoggingDictConfigurator
    
    env = Environment[env.upper()] if env else ''
    
    if reset:
        clear_logging_handlers()
        
    config: PFundConfig = get_config()
    logging_config: dict = get_logging_config()

    log_path = config.log_path / env if env else config.log_path
    log_path.mkdir(parents=True, exist_ok=True)

    # ≈ logging.config.dictConfig(logging_config) with a custom configurator
    logging_configurator = LoggingDictConfigurator(
        log_path=log_path, 
        logging_config=logging_config, 
        lazy=True,
        use_colored_logger=True,
    )
    logging_configurator.configure()
    
    setup_exception_logging(logger_name=project_name)
    return logging_config


def get_config() -> PFundConfig:
    """Lazy singleton - only creates config when first called.
    Also loads the .env file.
    """
    global _config
    if _config is None:
        _config = PFundConfig()
    return _config


def get_logging_config() -> dict:
    global _logging_config
    if _logging_config is None:
        _logging_config = configure_logging()
    return _logging_config


def configure(
    data_path: str | None = None,
    log_path: str | None = None,
    cache_path: str | None = None,
    persist: bool = False,
) -> PFundConfig:
    '''
    Configures the global config object.
    Args:
        data_path: Path to the data directory.
        log_path: Path to the log directory.
        cache_path: Path to the cache directory.
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
            setattr(config, k, v)
    
    config.ensure_dirs()
    
    if persist:
        config.save()
        
    return config


def configure_logging(logging_config: dict | None=None, debug: bool=False) -> dict:
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
    def __init__(self):
        super().__init__(project_name=project_name, source_file=__file__)
        # TODO: when mtflow is ready
        # artifact_path: str | None = None,
        # TODO: integrate with pfund_kit Configuration, confirm their usage
        self.storage: DataStorage | None = None
        self.storage_options: dict | None = None
        self.use_deltalake: bool | None = None

    def _initialize_from_data(self):
        """No additional config attributes to initialize."""
        pass
    
    # TODO: when compose.yml is in use
    def prepare_docker_context(self):
        pass
        # import os
        # os.environ['PFUND_DATA_PATH'] = self.data_path  # used in docker-compose.yml
    
    # FIXME: add more @property methods for different paths, e.g. hub_path, strategy_path etc. remove get_path method
    # TODO: also update cli command "pfund clear data" to use the new paths
    # TODO: move to mtflow?
    # def get_path(
    #     self, 
    #     name: Union[
    #         Literal['config', 'cache', 'log', 'data'],
    #         Literal['backtest', 'hub', 'template'],
    #         Literal['strategy', 'model', 'feature', 'indicator'],
    #         Literal['notebook', 'dashboard'],
    #     ]
    # ) -> Path:
    #     if name == 'config':
    #         return CONFIG_FILE_PATH
    #     elif name in ['cache', 'log', 'data']:
    #         return getattr(self, f'{name}_path')
    #     elif name in ['backtest', 'template']:
    #         return self.data_path / f'{name}s'
    #     elif name == 'hub':
    #         return self.data_path / 'hub'
    #     elif name == 'strategy':
    #         return self.data_path / 'hub' / 'strategies'
    #     elif name in ['model', 'feature', 'indicator']:
    #         return self.data_path / 'hub' / f'{name}s'
    #     elif name in ['notebook', 'dashboard']:
    #         return self.data_path / 'templates' / f'{name}s'
    #     else:
    #         raise ValueError(f'Invalid path name: {name}')