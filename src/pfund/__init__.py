from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # need these imports to support IDE hints:

    from pfund_blocks.entities.venues.venue_base import VenueConfig
    from pfund_bybit import Bybit as Bybit

    # from pfund_hyperliquid import Hyperliquid
    # from pfund_binance import Binance
    # from pfund_okx import OKX
    import pfund.hub as hub
    from pfund.backtest.cv.cross_validation import CrossValidation
    from pfund.backtest.cv.holdout import Holdout
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
    from pfund.data_config import DataConfig
    from pfund.engines.backtest_engine import BacktestEngine
    from pfund.engines.settings.backtest_engine_settings import BacktestEngineSettings
    from pfund.engines.settings.trade_engine_settings import TradeEngineSettings
    from pfund.engines.trade_engine import TradeEngine
    from pfund.simulated.simulated_account import SimulatedAccount
    from pfund.typing import BacktestDataFrame

from pfund.config import configure, configure_logging, get_config


def __getattr__(name: str):
    if name == "__version__":
        from importlib.metadata import version

        return version("pfund")
    elif name == "hub":
        import pfund.hub as hub

        return hub
    elif name == "VenueConfig":
        from pfund_blocks.entities.venues.venue_base import VenueConfig

        return VenueConfig
    elif name == "BacktestEngine":
        from pfund.engines.backtest_engine import BacktestEngine

        return BacktestEngine
    elif name == "Holdout":
        from pfund.backtest.cv.holdout import Holdout

        return Holdout
    elif name == "CrossValidation":
        from pfund.backtest.cv.cross_validation import CrossValidation

        return CrossValidation
    elif name == "DataConfig":
        from pfund.data_config import DataConfig

        return DataConfig
    elif name == "BacktestDataFrame":
        from pfund.typing import BacktestDataFrame

        return BacktestDataFrame
    elif name == "SimulatedAccount":
        from pfund.simulated.simulated_account import SimulatedAccount

        return SimulatedAccount
    elif name == "TradeEngine":
        from pfund.engines.trade_engine import TradeEngine

        return TradeEngine
    elif name == "TradeEngineSettings":
        from pfund.engines.settings.trade_engine_settings import TradeEngineSettings

        return TradeEngineSettings
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
    # ***TRADING VENUES***
    elif name == "Bybit":
        return _import_official_venue("pfund_bybit", "Bybit")
    # elif name == "Hyperliquid":
    #     return _import_official_venue("pfund_hyperliquid", "Hyperliquid")
    # elif name == "Binance":
    #     return _import_official_venue("pfund_binance", "Binance")
    # elif name == "OKX":
    #     return _import_official_venue("pfund_okx", "OKX")
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
    "__version__",
    "hub",
    "configure",
    "get_config",
    "configure_logging",
    "DataConfig",
    "VenueConfig",
    "SimulatedAccount",
    "Holdout",
    "CrossValidation",
    "BacktestDataFrame",
    # engines
    "BacktestEngine",
    "TradeEngine",
    "TradeEngineSettings",
    "BacktestEngineSettings",
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
)


def __dir__():
    return sorted(__all__)
