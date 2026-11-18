from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pfund.engines.settings.base_engine_settings import BaseEngineSettings

from pfund_blocks.enums.env import Environment

from pfund.engines.settings.backtest_engine_settings import BacktestEngineSettings
from pfund.engines.settings.trade_engine_settings import TradeEngineSettings

# keyed by str: settings.toml section names are plain strings (Environment is a StrEnum)
ENGINE_SETTINGS: dict[str, type[BaseEngineSettings]] = {
    Environment.BACKTEST: BacktestEngineSettings,
    Environment.PAPER: TradeEngineSettings,
    Environment.LIVE: TradeEngineSettings,
}

__all__ = [
    "ENGINE_SETTINGS",
    "BacktestEngineSettings",
    "TradeEngineSettings",
]
