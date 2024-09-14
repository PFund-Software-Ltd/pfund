import os
import sys
import importlib
import multiprocessing
import logging
from types import TracebackType
from dataclasses import dataclass

import yaml
# from rich.traceback import install

from pfund.const.paths import PROJ_NAME, LOG_PATH, MAIN_PATH, DATA_PATH, USER_CONFIG_FILE_PATH

# install(show_locals=False)  # rich will set its own sys.excepthook
# rich_excepthook = sys.excepthook  # get rich's excepthook


def _custom_excepthook(exception_class: type[BaseException], exception: BaseException, traceback: TracebackType):
    '''Catches any uncaught exceptions and logs them'''
    # sys.__excepthook__(exception_class, exception, traceback)
    try:
        # rich_excepthook(exception_class, exception, traceback)  # this will raise the exception to the console
        raise exception
    except:
        logging.getLogger(PROJ_NAME).exception('Uncaught exception:')
        
        
def dynamic_import(path: str):
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path) and '__pycache__' not in item_path:
            for type_ in ['strategies', 'models', 'features', 'indicators', 
                          'backtests', 'notebooks', 'spreadsheets', 'dashboards']:
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
class ConfigHandler:
    data_path: str = str(DATA_PATH)
    log_path: str = str(LOG_PATH)
    logging_config_file_path: str = f'{MAIN_PATH}/logging.yml'
    logging_config: dict | None = None
    use_fork_process: bool = True
    use_custom_excepthook: bool = True
    env_file_path: str | None=None
    debug: bool = False
    
    @classmethod
    def load_config(cls):
        config_file_path = USER_CONFIG_FILE_PATH
        if config_file_path.is_file():
            with open(config_file_path, 'r') as f:
                config = yaml.safe_load(f) or {}
        else:
            config = {}
        return cls(**config)
    
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
        self.logging_config = self.logging_config or {}
        
        for path in [self.strategy_path, self.model_path, self.feature_path, self.indicator_path,
                     self.backtest_path, self.notebook_path, self.spreadsheet_path, self.dashboard_path]:
            if not os.path.exists(path):
                os.makedirs(path)
                print(f'created {path}')
            sys.path.append(path)
            dynamic_import(path)
        
        if self.use_fork_process and sys.platform != 'win32':
            multiprocessing.set_start_method('fork', force=True)
        
        if self.use_custom_excepthook and sys.excepthook is sys.__excepthook__:
            sys.excepthook = _custom_excepthook
            
    def load_env_file(self, env_file_path: str | None):
        from dotenv import find_dotenv, load_dotenv
        
        if not env_file_path:
            found_env_file_path = find_dotenv(usecwd=True, raise_error_if_not_found=False)
            if found_env_file_path:
                print(f'.env file path is not specified, using env file in "{found_env_file_path}"')
            else:
                # print('.env file is not found')
                return
        load_dotenv(env_file_path, override=True)
    
    def enable_debug_mode(self):
        '''Enables debug mode by setting the log level to DEBUG for all stream handlers'''
        if 'handlers' not in self.logging_config:
            self.logging_config['handlers'] = {}
        for handler in ['stream_handler', 'stream_path_handler']:
            if handler not in self.logging_config['handlers']:
                self.logging_config['handlers'][handler] = {}
            self.logging_config['handlers'][handler]['level'] = 'DEBUG'
    

def configure(
    data_path: str = str(DATA_PATH),
    log_path: str = str(LOG_PATH),
    logging_config_file_path: str = f'{MAIN_PATH}/logging.yml',
    logging_config: dict | None=None,
    use_fork_process: bool=True,
    use_custom_excepthook: bool=False,
    env_file_path: str | None = None,
    debug: bool | None = None,
):
    return ConfigHandler(
        data_path=data_path,
        log_path=log_path,
        logging_config_file_path=logging_config_file_path,
        logging_config=logging_config,
        use_fork_process=use_fork_process,
        use_custom_excepthook=use_custom_excepthook,
        env_file_path=env_file_path,
        debug=debug,
    )