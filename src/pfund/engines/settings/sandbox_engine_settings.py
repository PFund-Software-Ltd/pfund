from decimal import Decimal

from pydantic import Field, field_validator

from pfund.engines.settings.trade_engine_settings import TradeEngineSettings
from pfund.enums import TradingVenue
from pfund.typing import Currency, ProductName


class SandboxEngineSettings(TradeEngineSettings):
    replay_mode: bool = Field(
        default=True,
        description="""
        Replay historical data as if it were live. No real venue connection is
        made when enabled. When disabled, connect to the real venue for live market
        data while continuing to use pfund's local fake server for bookkeeping.
        """,
    )
    replay_pace: float | None = Field(
        default=0.0,
        ge=0,
        description="""
        Seconds between row emissions in replay mode. Use 0 to replay as quickly as
        possible, a positive number for a fixed cadence, or None to follow the data's
        timestamps. Ignored when replay_mode is disabled.
        """,
    )
    initial_balances: dict[TradingVenue | str, dict[Currency, Decimal]] = Field(
        default_factory=dict
    )
    initial_positions: dict[TradingVenue | str, dict[ProductName, Decimal]] = Field(
        default_factory=dict
    )
    close_positions_at_end: bool = Field(
        default=True,
        description="""
        Close positions at the end of the simulation, i.e. engine.end()
        """,
    )

    @field_validator("initial_balances", "initial_positions", mode="before")
    @classmethod
    def _validate_initial_balances_and_positions(
        cls, v: dict[TradingVenue | str, dict[str, Decimal]]
    ):
        return {
            TradingVenue[venue.upper()]: {
                ccy_or_pdt.upper(): Decimal(amount)
                for ccy_or_pdt, amount in bal_or_pos.items()
            }
            for venue, bal_or_pos in v.items()
        }
