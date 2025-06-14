from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    # need these imports to support IDE hints:
    import pfund_plot as plot
    from pfund.const.aliases import ALIASES as aliases
    from pfund.engines.backtest_engine import BacktestEngine
    from pfund.engines.trade_engine import TradeEngine
    from pfund.strategies.strategy_base import BaseStrategy as Strategy
    from pfund.models.model_base import BaseModel as Model
    from pfund.models.pytorch_model import PytorchModel
    from pfund.models.sklearn_model import SklearnModel
    from pfund.features.feature_base import BaseFeature as Feature
    from pfund.indicators.indicator_base import BaseIndicator as Indicator
    from pfund.indicators.talib_indicator import TalibIndicator
    from pfund.indicators.ta_indicator import TaIndicator
    from pfund.brokers.broker_crypto import CryptoBroker
    from pfund.brokers.broker_dapp import DappBroker
    from pfund.brokers.ib.broker_ib import (
        IBBroker,
        IBBroker as IB,
    )
    from pfund.exchanges import Bybit

import sys
from importlib.metadata import version

from rich.console import Console

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
    elif name == 'plot':
        import pfund_plot as plot
        return plot
    elif name == 'BacktestEngine':
        from pfund.engines.backtest_engine import BacktestEngine
        return BacktestEngine
    elif name == 'TradeEngine':
        from pfund.engines.trade_engine import TradeEngine
        return TradeEngine
    elif name == "Strategy":
        from pfund.strategies.strategy_base import BaseStrategy as Strategy
        return Strategy
    elif name == "Model":
        from pfund.models.model_base import BaseModel as Model
        return Model
    elif name == "Feature":
        from pfund.features.feature_base import BaseFeature as Feature
        return Feature
    elif name == "PytorchModel":
        from pfund.models.pytorch_model import PytorchModel
        return PytorchModel
    elif name == "SklearnModel":
        from pfund.models.sklearn_model import SklearnModel
        return SklearnModel
    elif name == "Indicator":
        from pfund.indicators.indicator_base import BaseIndicator as Indicator
        return Indicator
    elif name == "TalibIndicator":
        from pfund.indicators.talib_indicator import TalibIndicator
        return TalibIndicator
    elif name == "TaIndicator":
        from pfund.indicators.ta_indicator import TaIndicator
        return TaIndicator
    elif name == "CryptoBroker":
        from pfund.brokers.broker_crypto import CryptoBroker
        return CryptoBroker
    elif name in ("IBBroker", 'IB'):
        from pfund.brokers.ib.broker_ib import IBBroker
        return IBBroker
    elif name == "DappBroker":
        from pfund.brokers.broker_dapp import DappBroker
        return DappBroker
    elif name == "Bybit":
        from pfund.exchanges.bybit.exchange import Exchange
        return Exchange
    elif name == "Binance":
        from pfund.exchanges.binance.exchange import Exchange
        return Exchange
    elif name == "Okx":
        from pfund.exchanges.okx.exchange import Exchange
        return Exchange


print_error = lambda msg: print(f'\033[91m{msg}\033[0m')
print_warning = lambda msg: print(f'\033[93m{msg}\033[0m')
cprint = Console().print


__version__ = version('pfund')
__all__ = (
    '__version__',
    'configure',
    'get_config',
    'aliases',
    "what_is",
    'plot',
    'BacktestEngine',
    'TradeEngine',
    'Strategy',
    'Model',
    'Feature',
    'PytorchModel',
    'SklearnModel',
    'Indicator',
    'TalibIndicator',
    'TaIndicator',
    'CryptoBroker',
    'IBBroker',
    'DappBroker',
    'Bybit',
)
def __dir__():
    return sorted(__all__)
