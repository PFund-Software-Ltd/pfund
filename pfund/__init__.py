from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    # need these imports to support IDE hints:
    import pfund_plot as plot
    from pfund.aliases import ALIASES as alias
    from pfund.engines.backtest_engine import BacktestEngine
    from pfund.engines.trade_engine import TradeEngine
    from pfund.strategies.strategy_base import BaseStrategy as Strategy
    from pfund.models.model_base import BaseModel as Model
    from pfund.models.pytorch_model import PytorchModel
    from pfund.models.sklearn_model import SklearnModel
    from pfund.features.feature_base import BaseFeature as Feature
    from pfund.indicators.indicator_base import BaseIndicator as Indicator
    from pfund.indicators.talib_indicator import TalibIndicator
    from pfund.brokers.broker_crypto import CryptoBroker
    from pfund.brokers.broker_defi import DeFiBroker
    from pfund.brokers.interactive_brokers.broker import (
        InteractiveBrokers as IBKR,
        InteractiveBrokers as IB,
    )
    from pfund.exchanges import Bybit

from importlib.metadata import version

from pfund.config import get_config, configure, configure_logging


def what_is(alias: str) -> str:
    from pfund.aliases import ALIASES
    return ALIASES.resolve(alias)


def __getattr__(name: str):
    if name == 'alias':
        from pfund.aliases import ALIASES
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
    elif name == "CryptoBroker":
        from pfund.brokers.broker_crypto import CryptoBroker
        return CryptoBroker
    elif name in ("InteractiveBrokers", "IBKR", "IB"):
        from pfund.brokers.interactive_brokers.broker import InteractiveBrokers
        return InteractiveBrokers
    elif name == "DeFiBroker":
        from pfund.brokers.broker_defi import DeFiBroker
        return DeFiBroker
    elif name == "Bybit":
        from pfund.exchanges import Bybit
        return Bybit
    elif name == "Binance":
        from pfund.exchanges import Binance
        return Binance
    elif name.upper() == 'OKX':
        from pfund.exchanges import OKX
        return OKX
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


print_error = lambda msg: print(f'\033[91m{msg}\033[0m')
print_warning = lambda msg: print(f'\033[93m{msg}\033[0m')


__version__ = version('pfund')
__all__ = (
    '__version__',
    'configure',
    'get_config',
    'configure_logging',
    'print_error',
    'print_warning',
    'alias',
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
    'IBKR', 'IB',
    'CryptoBroker',
    'DeFiBroker',
    'Bybit',
    'Binance',
    'OKX',
)
def __dir__():
    return sorted(__all__)
