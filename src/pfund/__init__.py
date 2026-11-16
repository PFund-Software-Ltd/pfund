from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # need these imports to support IDE hints:

    from pfund_bybit import Bybit

    # from pfund_hyperliquid import Hyperliquid
    # from pfund_binance import Binance
    # from pfund_okx import OKX
    import pfund.hub as hub
    from pfund.backtest.cv.cross_validation import CrossValidation
    from pfund.backtest.cv.holdout import Holdout
    from pfund.backtest.typing import PandasBacktestDataFrame, PolarsBacktestDataFrame
    from pfund.components.features.feature_base import BaseFeature as Feature
    from pfund.components.features.feature_talib import (
        TALibIndicator,
        TALibIndicator as Indicator,
        TALibIndicator as TalibIndicator,
    )
    from pfund.components.models.model_base import BaseModel as Model
    from pfund.components.models.pytorch_model import (
        PyTorchModel,
        PyTorchModel as PytorchModel,
    )
    from pfund.components.models.sklearn_model import (
        SKLearnModel,
        SKLearnModel as SklearnModel,
    )
    from pfund.components.strategies.strategy_base import BaseStrategy as Strategy
    from pfund.engines.backtest_engine import BacktestEngine
    from pfund.engines.sandbox_engine import SandboxEngine
    from pfund.engines.settings.backtest_engine_settings import BacktestEngineSettings
    from pfund.engines.settings.sandbox_engine_settings import SandboxEngineSettings
    from pfund.engines.settings.trade_engine_settings import TradeEngineSettings
    from pfund.engines.trade_engine import TradeEngine

from pfund.config import configure, configure_logging, get_config


def __getattr__(name: str):
    if name == "__version__":
        from importlib.metadata import version

        return version("pfund")
    elif name == "hub":
        import pfund.hub as hub

        return hub
    elif name == "BacktestEngine":
        from pfund.engines.backtest_engine import BacktestEngine

        return BacktestEngine
    elif name == "Holdout":
        from pfund.backtest.cv.holdout import Holdout

        return Holdout
    elif name == "CrossValidation":
        from pfund.backtest.cv.cross_validation import CrossValidation

        return CrossValidation
    elif name == "PolarsBacktestDataFrame":
        from pfund.backtest.typing import PolarsBacktestDataFrame

        return PolarsBacktestDataFrame
    elif name == "PandasBacktestDataFrame":
        from pfund.backtest.typing import PandasBacktestDataFrame

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
    elif name in ("TALibIndicator", "TalibIndicator", "Indicator"):
        from pfund.components.features.feature_talib import TALibIndicator

        return TALibIndicator
    elif name == "Hyperliquid":
        return _import_official_venue("pfund_hyperliquid", "Hyperliquid")
    elif name == "Binance":
        return _import_official_venue("pfund_binance", "Binance")
    elif name == "Bybit":
        return _import_official_venue("pfund_bybit", "Bybit")
    elif name == "OKX":
        return _import_official_venue("pfund_okx", "OKX")
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def _import_official_venue(module: str, venue: str):
    """Imports an official trading venue class (e.g. pf.Bybit) from its plugin, e.g. module="pfund_bybit".

    The install hint points to pfund's extra (e.g. "pfund[bybit]"), which pins a verified plugin version.
    """
    from pfund_kit.plugins import import_official_plugin

    extra = module.removeprefix("pfund_")
    return import_official_plugin(
        module, venue, label=f"pf.{venue}", install=f'"pfund[{extra}]"'
    )


__all__ = (  # noqa: RUF022
    "hub",
    "configure",
    "get_config",
    "configure_logging",
    # cv
    "Holdout",
    "CrossValidation",
    # engines
    "BacktestEngine",
    "SandboxEngine",
    "TradeEngine",
    "TradeEngineSettings",
    "SandboxEngineSettings",
    "BacktestEngineSettings",
    # backtest dfs
    "PolarsBacktestDataFrame",
    "PandasBacktestDataFrame",
    # components
    "Strategy",
    "Model",
    "Feature",
    "PyTorchModel",
    "PytorchModel",
    "SKLearnModel",
    "SklearnModel",
    "Indicator",
    "TalibIndicator",
    "TALibIndicator",
    # trading venues
    "Bybit",
)


def __dir__():
    return sorted(__all__)
