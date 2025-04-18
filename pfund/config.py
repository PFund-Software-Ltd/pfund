from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund.typing import tSTORAGE
    from pfund.plogging.config import LoggingDictConfigurator
    
import os
import sys
from pathlib import Path
import importlib
import logging
import shutil
import importlib.resources
from types import TracebackType
from dataclasses import dataclass, asdict, field, MISSING

import yaml
# from rich.traceback import install

from pfund.const.paths import (
    PROJ_NAME, 
    LOG_PATH, 
    CONFIG_PATH, 
    DATA_PATH, 
    CACHE_PATH,
    CONFIG_FILE_PATH
)

__all__ = [
    'get_config',
    'configure',
]

# install(show_locals=False)  # rich will set its own sys.excepthook
# rich_excepthook = sys.excepthook  # get rich's excepthook


def _custom_excepthook(exception_class: type[BaseException], exception: BaseException, traceback: TracebackType):
    '''Catches any uncaught exceptions and logs them'''
    # sys.__excepthook__(exception_class, exception, traceback)
    logging.getLogger(PROJ_NAME).exception('Uncaught exception:', exc_info=(exception_class, exception, traceback))
        
        
def _dynamic_import(path: str):
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path) and '__pycache__' not in item_path:
            for type_ in ['strategies', 'models', 'features', 'indicators', 
                          'notebooks', 'spreadsheets', 'dashboards']:
                if type_ in path:
                    break
            else:
                raise Exception(f'Invalid {path=} for dynamic import')
            module_path = os.path.join(item_path, '__init__.py')
            if os.path.isfile(module_path):
                spec = importlib.util.spec_from_file_location(item, module_path)
                module = importlib.util.module_from_spec(spec)
                module_space_name = '.'.join(['pfund', type_, item])
                sys.modules[module_space_name] = module
                spec.loader.exec_module(module)  # load the module, __init__.py in this case
                print(f'dynamically imported {module} from {module_path}')
            else:
                print(f'__init__.py not found in {item_path}, import failed')


@dataclass
class Configuration:
    data_path: str = str(DATA_PATH)
    log_path: str = str(LOG_PATH)
    cache_path: str = str(CACHE_PATH)
    logging_config_file_path: str = f'{CONFIG_PATH}/logging.yml'
    docker_compose_file_path: str = f'{CONFIG_PATH}/docker-compose.yml'
    custom_excepthook: bool = True
    env_file_path: str = f'{CONFIG_PATH}/.env'
    debug: bool = False
    storage: tSTORAGE = 'local'
    storage_options: dict = field(default_factory=dict)

    # NOTE: without type annotation, they will NOT be treated as dataclass fields but as class attributes
    _logging_config = {}
    _logging_configurator = None
    _instance = None
    _verbose = False

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls.load()
        return cls._instance

    @classmethod
    def set_verbose(cls, verbose: bool):
        cls._verbose = verbose
    
    @classmethod
    def load(cls) -> Configuration:
        '''Loads user's config file and returns a Configuration object'''
        CONFIG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        # Create default config from dataclass fields
        default_config = {}
        for field in cls.__dataclass_fields__.values():
            if field.name.startswith('_'):  # Skip private fields
                continue
            if field.default_factory is not MISSING:
                default_config[field.name] = field.default_factory()
            else:
                default_config[field.name] = field.default
                
        needs_update = False
        if CONFIG_FILE_PATH.is_file():
            with open(CONFIG_FILE_PATH, 'r') as f:
                saved_config = yaml.safe_load(f) or {}
                if cls._verbose:
                    print(f"{PROJ_NAME} config loaded from {CONFIG_FILE_PATH}.")
                # Check for new or removed fields
                new_fields = set(default_config.keys()) - set(saved_config.keys())
                removed_fields = set(saved_config.keys()) - set(default_config.keys())
                needs_update = bool(new_fields or removed_fields)
                
                if cls._verbose and needs_update:
                    if new_fields:
                        print(f"New config fields detected: {new_fields}")
                    if removed_fields:
                        print(f"Removed config fields detected: {removed_fields}")
                        
                # Filter out removed fields and merge with defaults
                saved_config = {k: v for k, v in saved_config.items() if k in default_config}
                config = {**default_config, **saved_config}
        else:
            config = default_config
            needs_update = True
        config = cls(**config)
        if needs_update:
            config.dump()
        return config
    
    def dump(self):
        with open(CONFIG_FILE_PATH, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False)
            if self._verbose:
                print(f"{PROJ_NAME} config saved to {CONFIG_FILE_PATH}.")
    
    @property
    def logging_config(self):
        return self._logging_config
    
    @logging_config.setter
    def logging_config(self, value: dict):
        self._logging_config = value
    
    @property
    def strategy_path(self):
        return f'{self.data_path}/hub/strategies'
    
    @property
    def model_path(self):
        return f'{self.data_path}/hub/models'
    
    @property
    def feature_path(self):
        return f'{self.data_path}/hub/features'
    
    @property
    def indicator_path(self):
        return f'{self.data_path}/hub/indicators'
    
    @property
    def backtest_path(self):
        return f'{self.data_path}/backtests'
    
    @property
    def notebook_path(self):
        return f'{self.data_path}/templates/notebooks'
    
    @property
    def spreadsheet_path(self):
        return f'{self.data_path}/templates/spreadsheets'
    
    @property
    def dashboard_path(self):
        return f'{self.data_path}/templates/dashboards'
    
    @property
    def artifact_path(self):
        return f'{self.data_path}/artifacts'
    
    def __post_init__(self):
        self._initialize_files()
        self._initialize_configs()
        
    def _initialize_files(self):
        '''Creates .env and copy logging.yml and docker-compose.yml from package directory to the user config path'''
        package_dir = Path(importlib.resources.files(PROJ_NAME)).resolve().parents[0]
        for path in [self.env_file_path, self.logging_config_file_path, self.docker_compose_file_path]:
            path = Path(path)
            try:
                if not path.exists():
                    if path.name == '.env':
                        path.touch(exist_ok=True)
                    else:
                        shutil.copy(package_dir / path.name, CONFIG_PATH)
            except Exception as e:
                print(f'Error creating or copying {path.name}: {e}')
        
    def _initialize_configs(self):
        for path in [
            self.cache_path, self.log_path,
            self.strategy_path, self.model_path, self.feature_path, self.indicator_path,
            self.backtest_path, self.notebook_path, self.spreadsheet_path, self.dashboard_path, 
            self.artifact_path,
        ]:
            if not os.path.exists(path):
                os.makedirs(path)
                if self._verbose:
                    print(f'created {path}')
            sys.path.append(path)
            if path not in [self.backtest_path, self.cache_path, self.log_path]:
                _dynamic_import(path)
        
        if self.custom_excepthook and sys.excepthook is sys.__excepthook__:
            sys.excepthook = _custom_excepthook
        
        self.load_env_file(self.env_file_path)
        
        if self.debug:
            self.enable_debug_mode()
        
    def load_env_file(self, env_file_path: str=''):
        from dotenv import find_dotenv, load_dotenv
        if not env_file_path:
            env_file_path = find_dotenv(usecwd=True, raise_error_if_not_found=False)
        
        if env_file_path:
            load_dotenv(env_file_path, override=True)
            if self._verbose:
                print(f'{PROJ_NAME} .env file loaded from {env_file_path}')
        else:
            if self._verbose:
                print(f'{PROJ_NAME} .env file is not found')
            return
    
    def enable_debug_mode(self):
        '''Enables debug mode by setting the log level to DEBUG for all stream handlers'''
        is_loggers_set_up = bool(logging.getLogger(PROJ_NAME).handlers)
        if is_loggers_set_up:
            if self._verbose:
                print('loggers are already set up, ignoring debug mode')
            return
        if 'handlers' not in self.logging_config:
            self.logging_config['handlers'] = {}
        for handler in ['stream_handler', 'stream_path_handler']:
            if handler not in self.logging_config['handlers']:
                self.logging_config['handlers'][handler] = {}
            self.logging_config['handlers'][handler]['level'] = 'DEBUG'
    
    def set_logging_configurator(self, logging_configurator: LoggingDictConfigurator):
        self._logging_configurator = logging_configurator
    

def configure(
    data_path: str | None = None,
    log_path: str | None = None,
    logging_config_file_path: str | None = None,
    logging_config: dict | None = None,
    docker_compose_file_path: str | None = None,
    env_file_path: str | None = None,
    custom_excepthook: bool | None = None,
    debug: bool | None = None,
    storage: tSTORAGE | None = None,
    storage_options: dict | None = None,
    verbose: bool = False,
    write: bool = False,
):
    '''Configures the global config object.
    It will override the existing config values from the existing config file or the default values.
    Args:
        write: If True, the config will be saved to the config file.
    '''
    NON_CONFIG_KEYS = ['verbose', 'write']
    config_updates = locals()
    for k in NON_CONFIG_KEYS:
        config_updates.pop(k)
    config_updates.pop('NON_CONFIG_KEYS')

    config = get_config(verbose=verbose)

    # Apply updates for non-None values
    for k, v in config_updates.items():
        if v is not None:
            setattr(config, k, v)
            
    if write:
        config.dump()
        
    config._initialize_configs()
    return config


def get_config(verbose: bool = False) -> Configuration:
    Configuration.set_verbose(verbose)
    return Configuration.get_instance()