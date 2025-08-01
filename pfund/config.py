from __future__ import annotations
from typing import TYPE_CHECKING, Literal, Union
if TYPE_CHECKING:
    from pfeed._typing import tStorage
    from pfund._typing import tEnvironment
    
import os
import sys
from pathlib import Path
import importlib
import logging
import shutil
import importlib.resources
from types import TracebackType
from dataclasses import dataclass, asdict, field, MISSING

# from rich.traceback import install

from pfeed.enums import DataStorage
from pfund.enums import Environment
from pfund.utils.utils import load_yaml_file, dump_yaml_file
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
        
        
# FIXME: should move this to pfund-hub
# it enables dynamic import like this: from pfund.strategies import SomeStrategy
def _dynamic_import(path: str):
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path) and '__pycache__' not in item_path:
            for type_ in ['strategies', 'models', 'features', 'indicators', 
                          'notebooks', 'dashboards']:
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
    data_path: Path = DATA_PATH
    log_path: Path = LOG_PATH
    cache_path: Path = CACHE_PATH
    logging_config_file_path: Path = CONFIG_PATH / 'logging.yml'
    docker_compose_file_path: Path = CONFIG_PATH / 'docker-compose.yml'
    custom_excepthook: bool = True
    debug: bool = False
    storage: DataStorage = DataStorage.LOCAL
    storage_options: dict = field(default_factory=dict)
    use_deltalake: bool = False

    # NOTE: without type annotation, they will NOT be treated as dataclass fields but as class attributes
    _logging_config = {}
    _instance = None
    _verbose = False

    # REVIEW: this won't be needed if we use pydantic.BaseModel instead of dataclass
    def _enforce_types(self):
        config_dict = asdict(self)
        for k, v in config_dict.items():
            _field = self.__dataclass_fields__[k]
            if _field.type == 'Path' and isinstance(v, str):
                setattr(self, k, Path(v))
            elif _field.type == 'DataStorage' and isinstance(v, str):
                setattr(self, k, DataStorage[v.upper()])

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls.load()
        return cls._instance

    @classmethod
    def set_verbose(cls, verbose: bool):
        cls._verbose = verbose
    
    @classmethod
    def _load_env_file(cls, env: Environment | tEnvironment):
        from dotenv import find_dotenv, load_dotenv
        env = Environment[env.upper()]
        filename = f'.env.{env.lower()}'
        env_file_path = find_dotenv(filename=filename, usecwd=True, raise_error_if_not_found=False)
        if env_file_path:
            load_dotenv(env_file_path, override=True)
            if cls._verbose:
                print(f'{PROJ_NAME} {filename} file loaded from {env_file_path}')
        else:
            if cls._verbose:
                print(f'{PROJ_NAME} {filename} file is not found')
    
    @classmethod
    def load(cls) -> Configuration:
        '''Loads user's config file and returns a Configuration object'''
        CONFIG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        # Create default config from dataclass fields
        default_config = {}
        for _field in cls.__dataclass_fields__.values():
            if _field.name.startswith('_'):  # Skip private fields
                continue
            if _field.default_factory is not MISSING:
                default_config[_field.name] = _field.default_factory()
            else:
                default_config[_field.name] = _field.default
                
        needs_update = False
        if CONFIG_FILE_PATH.is_file():
            current_config = load_yaml_file(CONFIG_FILE_PATH)
            if cls._verbose:
                print(f"Loaded {CONFIG_FILE_PATH}")
            # Check for new or removed fields
            new_fields = set(default_config.keys()) - set(current_config.keys())
            removed_fields = set(current_config.keys()) - set(default_config.keys())
            needs_update = bool(new_fields or removed_fields)
            
            if cls._verbose and needs_update:
                if new_fields:
                    print(f"New config fields detected: {new_fields}")
                if removed_fields:
                    print(f"Removed config fields detected: {removed_fields}")
                    
            # Filter out removed fields and merge with defaults
            current_config = {k: v for k, v in current_config.items() if k in default_config}
            config = {**default_config, **current_config}
        else:
            config = default_config
            needs_update = True
        config = cls(**config)
        if needs_update:
            config.dump()
        return config
    
    def dump(self):
        dump_yaml_file(CONFIG_FILE_PATH, asdict(self))
        if self._verbose:
            print(f"Created {CONFIG_FILE_PATH}")
    
    @property
    def logging_config(self):
        return self._logging_config
    
    @logging_config.setter
    def logging_config(self, value: dict):
        self._logging_config = value
    
    def get_path(
        self, 
        name: Union[
            Literal['config', 'cache', 'log', 'data'],
            Literal['backtest', 'hub', 'template'],
            Literal['strategy', 'model', 'feature', 'indicator'],
            Literal['notebook', 'dashboard'],
        ]
    ) -> Path:
        if name == 'config':
            return CONFIG_FILE_PATH
        elif name in ['cache', 'log', 'data']:
            return getattr(self, f'{name}_path')
        elif name in ['backtest', 'template']:
            return self.data_path / f'{name}s'
        elif name == 'hub':
            return self.data_path / 'hub'
        elif name == 'strategy':
            return self.data_path / 'hub' / 'strategies'
        elif name in ['model', 'feature', 'indicator']:
            return self.data_path / 'hub' / f'{name}s'
        elif name in ['notebook', 'dashboard']:
            return self.data_path / 'templates' / f'{name}s'
        else:
            raise ValueError(f'Invalid path name: {name}')
    
    def __post_init__(self):
        self._initialize()
        
    def _initialize(self):
        self._enforce_types()
        self._initialize_files()
        self._initialize_file_paths()
        if self.custom_excepthook and sys.excepthook is sys.__excepthook__:
            sys.excepthook = _custom_excepthook
        if self.debug:
            self.enable_debug_mode()
        
    def _initialize_files(self):
        '''Copies logging.yml and docker-compose.yml from package directory to the user config path'''
        package_dir = Path(importlib.resources.files(PROJ_NAME)).resolve().parents[0]
        for path in [self.logging_config_file_path, self.docker_compose_file_path]:
            if path.exists():
                continue
            try:
                filename = path.name
                # copies the file from site-packages/pfund to the user config path
                shutil.copy(package_dir / filename, CONFIG_PATH)
                print(f'Created {filename} in {CONFIG_PATH}')
            except Exception as e:
                print(f'Error creating or copying {path.name}: {e}')
        
    def _initialize_file_paths(self):
        for path in [self.cache_path, self.log_path, self.data_path]:
            if not os.path.exists(path):
                os.makedirs(path)
                if self._verbose:
                    print(f'{PROJ_NAME} created {path}')
    
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
    

def configure(
    data_path: str | None = None,
    log_path: str | None = None,
    cache_path: str | None = None,
    # TODO: when mtflow is ready
    # artifact_path: str | None = None,  
    logging_config_file_path: str | None = None,
    docker_compose_file_path: str | None = None,
    logging_config: dict | None = None,
    custom_excepthook: bool | None = None,
    debug: bool | None = None,
    storage: tStorage | None = None,
    storage_options: dict | None = None,
    use_deltalake: bool | None = None,
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

    Configuration.set_verbose(verbose)
    config = get_config()

    # Apply updates for non-None values
    for k, v in config_updates.items():
        if v is not None:
            setattr(config, k, v)
            
    if write:
        config.dump()
        
    config._initialize()
    return config


def get_config() -> Configuration:
    return Configuration.get_instance()