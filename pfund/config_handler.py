import os
import sys
import importlib
import multiprocessing
import logging
from types import TracebackType
from dataclasses import dataclass

# from rich.traceback import install

from pfund.const.paths import PROJ_NAME, PROJ_PATH, LOG_PATH, PROJ_CONFIG_PATH, DATA_PATH
# add python path so that for files like "ibapi" (official python code from IB)
# can find their modules
sys.path.append(f'{PROJ_PATH}/externals')

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
        
        
def import_strategies_models_features_or_indicators(path: str):
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path) and '__pycache__' not in item_path:
            for type_ in ['strategies', 'models', 'features', 'indicators']:
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
    logging_config_file_path: str = f'{PROJ_CONFIG_PATH}/logging.yml'
    logging_config: dict | None = None
    use_fork_process: bool = True
    use_custom_excepthook: bool = True
    
    def __post_init__(self):
        self.logging_config = self.logging_config or {}
        
        strategy_path, model_path = f'{self.data_path}/strategies', f'{self.data_path}/models'
        feature_path, indicator_path = f'{self.data_path}/features', f'{self.data_path}/indicators'
        for path in [strategy_path, model_path, feature_path, indicator_path]:
            if not os.path.exists(path):
                os.makedirs(path)
                print(f'created {path}')
            sys.path.append(path)
            import_strategies_models_features_or_indicators(path)
        
        if self.use_fork_process and sys.platform != 'win32':
            multiprocessing.set_start_method('fork', force=True)
        
        if self.use_custom_excepthook:
            sys.excepthook = _custom_excepthook
            

def configure(
    data_path: str = str(DATA_PATH),
    log_path: str = str(LOG_PATH),
    logging_config_file_path: str = f'{PROJ_CONFIG_PATH}/logging.yml',
    logging_config: dict | None=None,
    use_fork_process: bool=True,
    use_custom_excepthook: bool=True,
):
    return ConfigHandler(
        data_path=data_path,
        log_path=log_path,
        logging_config_file_path=logging_config_file_path,
        logging_config=logging_config,
        use_fork_process=use_fork_process,
        use_custom_excepthook=use_custom_excepthook,
    )