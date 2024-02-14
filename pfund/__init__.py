from importlib.metadata import version

from pfund.config_handler import configure
from pfund.engines import BacktestEngine, TrainEngine, TestEngine, TradeEngine
from pfund.strategies import Strategy
from pfund.models import Feature, Model
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
    'configure',
    'BacktestEngine', 'TrainEngine', 'TestEngine', 'TradeEngine',
    'Strategy', 'Model', 'PyTorchModel', 'SKLearnModel',
    'Feature', 'TAIndicator', 'TALibIndicator',
)
