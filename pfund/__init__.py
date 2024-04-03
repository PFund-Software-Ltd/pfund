import sys
from importlib.metadata import version

from pfund.const.paths import PROJ_PATH
# add python path so that for files like "ibapi" (official python code from IB) can find their modules
sys.path.append(f'{PROJ_PATH}/externals')
from pfund.config_handler import configure
from pfund.engines import BacktestEngine, TrainEngine, TestEngine, TradeEngine
from pfund.strategies import Strategy
from pfund.models import Feature, Model
from pfund.utils.aliases import ALIASES
try:
    from pfund.models import PyTorchModel
except ImportError:
    pass
try:
    from pfund.models import SKLearnModel
except ImportError:
    pass
from pfund.indicators import TAIndicator, TALibIndicator


__version__ = version('pfund')


__all__ = (
    '__version__',
    'configure', 'ALIASES',
    'BacktestEngine', 'TrainEngine', 'TestEngine', 'TradeEngine',
    'Strategy', 'Model', 'PyTorchModel', 'SKLearnModel',
    'Feature', 'TAIndicator', 'TALibIndicator',
)
