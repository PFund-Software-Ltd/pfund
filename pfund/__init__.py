import sys
from importlib.metadata import version

from pfund.config_handler import configure
from pfund.const.aliases import ALIASES as aliases
from pfund.const.paths import PROJ_PATH
# add python path so that for files like "ibapi" (official python code from IB) can find their modules
sys.path.append(f'{PROJ_PATH}/externals')
from pfund.engines import BacktestEngine, TradeEngine, TrainEngine, SandboxEngine
from pfund.strategies import Strategy
from pfund.models import Model, Feature, PytorchModel, SklearnModel
from pfund.indicators import Indicator, TalibIndicator, TaIndicator


def what_is(alias: str) -> str | None:
    if alias in aliases or alias.upper() in aliases:
        return aliases.get(alias, aliases.get(alias.upper(), None))


print_error = lambda msg: print(f'\033[91m{msg}\033[0m')
print_warning = lambda msg: print(f'\033[93m{msg}\033[0m')



__version__ = version('pfund')
__all__ = (
    '__version__',
    'configure', 
    'aliases',
    'BacktestEngine', 
    'TradeEngine',
    'TrainEngine', 
    'SandboxEngine',
    'Strategy', 
    'Model',
    'Feature',
    'PytorchModel',
    'SklearnModel',
    'Indicator',
    'TalibIndicator',
    'TaIndicator',
)
