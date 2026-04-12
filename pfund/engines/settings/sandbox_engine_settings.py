from decimal import Decimal

from pydantic import Field

from pfund.typing import Currency, ProductName
from pfund.enums import TradingVenue
from pfund.engines.settings.trade_engine_settings import TradeEngineSettings


class SandboxEngineSettings(TradeEngineSettings):
    initial_balances: dict[TradingVenue | str, dict[Currency, Decimal]] = Field(default_factory=dict)
    initial_positions: dict[TradingVenue | str, dict[ProductName, Decimal]] = Field(default_factory=dict)