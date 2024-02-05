import os
import sys
import importlib
from pathlib import Path
import multiprocessing
import logging
from types import TracebackType
from dataclasses import dataclass

# from rich.traceback import install

from pfund.const.paths import PROJ_NAME, PROJ_PATH, LOG_PATH, CONFIG_PATH, STRATEGY_PATH, MODEL_PATH, FEATURE_PATH, INDICATOR_PATH
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
        
        
def import_strategies_models_features_or_indicators(path: Path):
    path = str(path)
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
class Config:
    strategy_path: Path = STRATEGY_PATH
    model_path: Path = MODEL_PATH
    feature_path: Path = FEATURE_PATH
    indicator_path: Path = INDICATOR_PATH
    log_path: Path = LOG_PATH
    logging_config_file_path: Path = CONFIG_PATH / 'logging.yml'
    logging_config: dict | None = None
    use_fork_process: bool = True
    use_custom_excepthook: bool = True
    
    def __post_init__(self):
        for path in (self.strategy_path, self.model_path, self.feature_path, self.indicator_path):
            if not path.exists():
                os.makedirs(path)
                print(f'created {str(path)}')
            sys.path.append(path)
            import_strategies_models_features_or_indicators(path)
        
        if self.use_fork_process and sys.platform != 'win32':
            multiprocessing.set_start_method('fork', force=True)
        
        if self.use_custom_excepthook:
            sys.excepthook = _custom_excepthook
            

def configure(
    strategy_path: str | Path=STRATEGY_PATH,
    model_path: str | Path=MODEL_PATH,
    feature_path: str | Path=FEATURE_PATH,
    indicator_path: str | Path=INDICATOR_PATH,
    log_path: str | Path=LOG_PATH,
    logging_config_file_path: str | Path = CONFIG_PATH / 'logging.yml',
    logging_config: dict | None=None,
    use_fork_process: bool=True,
    use_custom_excepthook: bool=True,
):
    logging_config_file_path = Path(logging_config_file_path)
    assert logging_config_file_path.is_file(), f'{logging_config_file_path=} is not a file'
    return Config(
        strategy_path=Path(strategy_path),
        model_path=Path(model_path),
        feature_path=Path(feature_path),
        indicator_path=Path(indicator_path),
        log_path=Path(log_path),
        logging_config_file_path=logging_config_file_path,
        logging_config=logging_config,
        use_fork_process=use_fork_process,
        use_custom_excepthook=use_custom_excepthook,
    )