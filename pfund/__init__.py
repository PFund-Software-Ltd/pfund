from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    # need these imports to support IDE hints:
    aliases = ...
    from pfund.engines import (
        BacktestEngine, 
        TradeEngine, 
        TrainEngine, 
        SandboxEngine
    )
    from pfund.strategies import Strategy
    from pfund.models import (
        Model, 
        Feature, 
        PytorchModel, 
        SklearnModel
    )
    from pfund.indicators import (
        Indicator, 
        TalibIndicator, 
        TaIndicator
    )
    from pfund.brokers import (
        CryptoBroker, 
        IB_Broker
    )
    from pfund.exchanges import (
        BybitExchange,
        BinanceExchange,
        OkxExchange,
    )

import sys
from importlib.metadata import version

from pfund.config import get_config, configure
# FIXME: install by pfund-ibapi
from pfund.const.paths import PROJ_PATH
# add python path so that for files like "ibapi" (official python code from IB) can find their modules
sys.path.append(f'{PROJ_PATH}/externals')


def what_is(alias: str) -> str | None:
    from pfund.const.aliases import ALIASES
    if alias in ALIASES or alias.upper() in ALIASES:
        return ALIASES.get(alias, ALIASES.get(alias.upper(), None))


def __getattr__(name: str):
    if name == 'aliases':
        from pfund.const.aliases import ALIASES
        return ALIASES
    elif name == 'BacktestEngine':
        from pfund.engines import BacktestEngine
        return BacktestEngine
    elif name == 'TradeEngine':
        from pfund.engines import TradeEngine
        return TradeEngine
    elif name == 'TrainEngine':
        from pfund.engines import TrainEngine
        return TrainEngine
    elif name == 'SandboxEngine':
        from pfund.engines import SandboxEngine
        return SandboxEngine
    elif name == "Strategy":
        from pfund.strategies import Strategy
        return Strategy
    elif name == "Model":
        from pfund.models import Model
        return Model
    elif name == "Feature":
        from pfund.models import Feature
        return Feature
    elif name == "PytorchModel":
        from pfund.models import PytorchModel
        return PytorchModel
    elif name == "SklearnModel":
        from pfund.models import SklearnModel
        return SklearnModel
    elif name == "Indicator":
        from pfund.indicators import Indicator
        return Indicator
    elif name == "TalibIndicator":
        from pfund.indicators import TalibIndicator
        return TalibIndicator
    elif name == "TaIndicator":
        from pfund.indicators import TaIndicator
        return TaIndicator
    elif name == "CryptoBroker":
        from pfund.brokers import CryptoBroker
        return CryptoBroker
    elif name == "IB_Broker":
        from pfund.brokers import IB_Broker
        return IB_Broker
    elif name == "BybitExchange":
        from pfund.exchanges import BybitExchange
        return BybitExchange
    elif name == "BinanceExchange":
        from pfund.exchanges import BinanceExchange
        return BinanceExchange
    elif name == "OkxExchange":
        from pfund.exchanges import OkxExchange
        return OkxExchange


print_error = lambda msg: print(f'\033[91m{msg}\033[0m')
print_warning = lambda msg: print(f'\033[93m{msg}\033[0m')


__version__ = version('pfund')
__all__ = (
    '__version__',
    'configure',
    'get_config',
    'aliases',
    "what_is",
    # TODO:
    ...
)
