import sys
import multiprocessing
from types import TracebackType

from rich.console import Console
# from rich.traceback import install

from pfund.const.paths import PROJ_PATH
# add python path so that for files like "ibapi" (official python code from IB)
# can find their modules
sys.path.append(f'{PROJ_PATH}/externals')
from pfund.engines import *
from pfund.strategies import Strategy
from pfund.models import Feature, Model, PyTorchModel, SKLearnModel
from pfund.indicators import TAIndicator, TALibIndicator


cprint = Console().print
# install(show_locals=False)  # rich will set its own sys.excepthook
# rich_excepthook = sys.excepthook  # get rich's excepthook


def _custom_excepthook(exception_class: type[BaseException], exception: BaseException, traceback: TracebackType):
    '''Catches any uncaught exceptions and logs them'''
    import logging
    # sys.__excepthook__(exception_class, exception, traceback)
    try:
        # this will raise the exception to the console
        # rich_excepthook(exception_class, exception, traceback)
        raise exception
    except:
        logging.getLogger('pfund').exception('Uncaught exception:')


sys.excepthook = _custom_excepthook
multiprocessing.set_start_method('fork', force=True)


excludes = (
    'sys',
    'multiprocessing', 
    'TracebackType',
    'Console',
    'PROJ_PATH',
)
__all__ = [name for name in dir() if name not in excludes and not name.startswith('_')]