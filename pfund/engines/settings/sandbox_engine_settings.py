from decimal import Decimal

from pydantic import Field, field_validator

from pfund.typing import Currency, ProductName
from pfund.enums import TradingVenue
from pfund.engines.settings.trade_engine_settings import TradeEngineSettings


class SandboxEngineSettings(TradeEngineSettings):
    initial_balances: dict[TradingVenue | str, dict[Currency, Decimal]] = Field(default_factory=dict)
    initial_positions: dict[TradingVenue | str, dict[ProductName, Decimal]] = Field(default_factory=dict)

    @field_validator('initial_balances', 'initial_positions', mode='before')
    @classmethod
    def _validate_initial_balances_and_positions(cls, v: dict[TradingVenue | str, dict[str, Decimal]]):
        return {
            TradingVenue[venue.upper()]: {
                ccy_or_pdt.upper(): Decimal(amount) for ccy_or_pdt, amount in bal_or_pos.items()
            }
            for venue, bal_or_pos in v.items()
        }
