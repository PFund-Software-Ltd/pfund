from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # need these imports to support IDE hints:
    import pfund_plot as plot
    import pfund.hub as hub

    from pfund.brokers.crypto.broker import CryptoBroker
    from pfund.brokers.crypto.exchanges import Bybit
    from pfund.brokers.ibkr.broker import (
        InteractiveBrokers as IB,
    )
    from pfund.brokers.ibkr.broker import (
        InteractiveBrokers as IBKR,
    )
    from pfund._backtest.typing import PolarsBacktestDataFrame, PandasBacktestDataFrame
    from pfund.components.features.feature_base import BaseFeature as Feature
    from pfund.components.indicators.indicator_base import BaseIndicator as Indicator
    from pfund.components.indicators.talib_indicator import TalibIndicator
    from pfund.components.models.model_base import BaseModel as Model
    from pfund.components.models.pytorch_model import PytorchModel
    from pfund.components.models.sklearn_model import SklearnModel
    from pfund.components.strategies.strategy_base import BaseStrategy as Strategy
    from pfund.datas.data_config import DataConfig
    from pfund.engines.backtest_engine import BacktestEngine
    from pfund.engines.sandbox_engine import SandboxEngine
    from pfund.engines.settings.backtest_engine_settings import BacktestEngineSettings
    from pfund.engines.settings.sandbox_engine_settings import SandboxEngineSettings
    from pfund.engines.settings.trade_engine_settings import TradeEngineSettings
    from pfund.engines.trade_engine import TradeEngine
    from pfund.utils.aliases import ALIASES as alias  # noqa: N811
    # from pfund.brokers.broker_defi import DeFiBroker

from pfund.config import configure, configure_logging, get_config


def __getattr__(name: str):
    if name == "__version__":
        from importlib.metadata import version

        return version("pfund")
    elif name == "alias":
        from pfund.utils.aliases import ALIASES

        return ALIASES
    elif name == "plot":
        import pfund_plot as plot

        return plot
    elif name == "hub":
        import pfund.hub as hub

        return hub
    elif name == "DataConfig":
        from pfund.datas.data_config import DataConfig

        return DataConfig
    elif name == "BacktestEngine":
        from pfund.engines.backtest_engine import BacktestEngine

        return BacktestEngine
    elif name == "PolarsBacktestDataFrame":
        from pfund._backtest.typing import PolarsBacktestDataFrame

        return PolarsBacktestDataFrame
    elif name == "PandasBacktestDataFrame":
        from pfund._backtest.typing import PandasBacktestDataFrame

        return PandasBacktestDataFrame
    elif name == "TradeEngine":
        from pfund.engines.trade_engine import TradeEngine

        return TradeEngine
    elif name == "SandboxEngine":
        from pfund.engines.sandbox_engine import SandboxEngine

        return SandboxEngine
    elif name == "TradeEngineSettings":
        from pfund.engines.settings.trade_engine_settings import TradeEngineSettings

        return TradeEngineSettings
    elif name == "SandboxEngineSettings":
        from pfund.engines.settings.sandbox_engine_settings import SandboxEngineSettings

        return SandboxEngineSettings
    elif name == "BacktestEngineSettings":
        from pfund.engines.settings.backtest_engine_settings import (
            BacktestEngineSettings,
        )

        return BacktestEngineSettings
    elif name == "Strategy":
        from pfund.components.strategies.strategy_base import BaseStrategy as Strategy

        return Strategy
    elif name == "Model":
        from pfund.components.models.model_base import BaseModel as Model

        return Model
    elif name == "Feature":
        from pfund.components.features.feature_base import BaseFeature as Feature

        return Feature
    elif name == "PytorchModel":
        from pfund.components.models.pytorch_model import PytorchModel

        return PytorchModel
    elif name == "SklearnModel":
        from pfund.components.models.sklearn_model import SklearnModel

        return SklearnModel
    elif name == "Indicator":
        from pfund.components.indicators.indicator_base import (
            BaseIndicator as Indicator,
        )

        return Indicator
    elif name == "TalibIndicator":
        from pfund.components.indicators.talib_indicator import TalibIndicator

        return TalibIndicator
    elif name == "CryptoBroker":
        from pfund.brokers.crypto.broker import CryptoBroker

        return CryptoBroker
    elif name in ("InteractiveBrokers", "IBKR", "IB"):
        from pfund.brokers.ibkr.broker import InteractiveBrokers

        return InteractiveBrokers
    # elif name == "DeFiBroker":
    #     from pfund.brokers.broker_defi import DeFiBroker

    #     return DeFiBroker
    elif name == "Bybit":
        from pfund.brokers.crypto.exchanges import Bybit

        return Bybit
    # elif name == "Binance":
    #     from pfund.brokers.crypto.exchanges import Binance

    #     return Binance
    # elif name.upper() == "OKX":
    #     from pfund.brokers.crypto.exchanges import OKX

    #     return OKX
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = (
    "alias",
    "plot",
    "hub",
    # configs
    "configure",
    "get_config",
    "configure_logging",
    "DataConfig",
    # engines
    "BacktestEngine",
    "TradeEngine",
    "SandboxEngine",
    "TradeEngineSettings",
    "SandboxEngineSettings",
    "BacktestEngineSettings",
    # backtest
    "PolarsBacktestDataFrame",
    "PandasBacktestDataFrame",
    # components
    "Strategy",
    "Model",
    "Feature",
    "PytorchModel",
    "SklearnModel",
    "Indicator",
    "TalibIndicator",
    # brokers
    "IBKR",
    "IB",
    "CryptoBroker",
    # "DeFiBroker",
    # exchanges
    "Bybit",
    # "Binance",
    # "OKX",
)


def __dir__():
    return sorted(__all__)
