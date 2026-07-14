from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # need these imports to support IDE hints:
    import pfund_plot as plot
    import pfund.hub as hub

    from pfund.venues.ibkr import (
        InteractiveBrokers,
        InteractiveBrokers as IB,
        InteractiveBrokers as IBKR,
    )
    from pfund.venues.hyperliquid import Hyperliquid
    from pfund.venues.bybit import Bybit
    from pfund.venues.binance import Binance
    from pfund.venues.okx import OKX
    from pfund._backtest.typing import PolarsBacktestDataFrame, PandasBacktestDataFrame
    from pfund.components.features.feature_base import BaseFeature as Feature
    from pfund.components.features.feature_talib import (
        TalibIndicator,
        TalibIndicator as Indicator,
    )
    from pfund.components.models.model_base import BaseModel as Model
    from pfund.components.models.pytorch_model import (
        PyTorchModel,
        PyTorchModel as PytorchModel,
    )
    from pfund.components.models.jax_model import (
        JAXModel,
        JAXModel as JaxModel,
    )
    from pfund.components.models.sklearn_model import (
        SKLearnModel,
        SKLearnModel as SklearnModel,
    )
    from pfund.components.strategies.strategy_base import BaseStrategy as Strategy
    from pfund.datas.data_config import DataConfig
    from pfund.engines.backtest_engine import BacktestEngine
    from pfund.engines.sandbox_engine import SandboxEngine
    from pfund.engines.settings.backtest_engine_settings import BacktestEngineSettings
    from pfund.engines.settings.sandbox_engine_settings import SandboxEngineSettings
    from pfund.engines.settings.trade_engine_settings import TradeEngineSettings
    from pfund.engines.trade_engine import TradeEngine
    from pfund.utils.aliases import ALIASES as alias  # noqa: N811

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
    elif name in ("PyTorchModel", "PytorchModel"):
        from pfund.components.models.pytorch_model import PyTorchModel

        return PyTorchModel
    elif name in ("SKLearnModel", "SklearnModel"):
        from pfund.components.models.sklearn_model import SKLearnModel

        return SKLearnModel
    elif name in ("JAXModel", "JaxModel"):
        from pfund.components.models.jax_model import JAXModel

        return JAXModel
    elif name in ("TalibIndicator", "Indicator"):
        from pfund.components.features.feature_talib import TalibIndicator

        return TalibIndicator
    elif name in ("InteractiveBrokers", "IBKR", "IB"):
        from pfund.venues.ibkr import InteractiveBrokers

        return InteractiveBrokers
    elif name == "Hyperliquid":
        from pfund.venues.hyperliquid import Hyperliquid

        return Hyperliquid
    elif name == "Binance":
        from pfund.venues.binance import Binance

        return Binance
    elif name == "Bybit":
        from pfund.venues.bybit import Bybit

        return Bybit
    elif name == "OKX":
        from pfund.venues.okx import OKX

        return OKX
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = (
    "alias",
    "plot",
    "hub",
    "configure",
    "get_config",
    "configure_logging",
    "DataConfig",
    "BacktestEngine",
    "TradeEngine",
    "SandboxEngine",
    "TradeEngineSettings",
    "SandboxEngineSettings",
    "BacktestEngineSettings",
    "PolarsBacktestDataFrame",
    "PandasBacktestDataFrame",
    "Strategy",
    "Model",
    "Feature",
    "PyTorchModel",
    "PytorchModel",
    "SKLearnModel",
    "SklearnModel",
    "JAXModel",
    "JaxModel",
    "Indicator",
    "TalibIndicator",
    "InteractiveBrokers",
    "IBKR",
    "IB",
    "Hyperliquid",
    "Binance",
    "Bybit",
    "OKX",
)


def __dir__():
    return sorted(__all__)
