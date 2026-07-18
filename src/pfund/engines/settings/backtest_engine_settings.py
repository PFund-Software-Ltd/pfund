from decimal import Decimal

from pydantic import Field, field_validator

from pfund.engines.settings.base_engine_settings import BaseEngineSettings
from pfund.enums import TradingVenue
from pfund.typing import Currency, ProductName


class BacktestEngineSettings(BaseEngineSettings):
    # If not provided, will use the default initial balances in SimulatedBroker
    initial_balances: dict[TradingVenue | str, dict[Currency, Decimal]] = Field(
        default_factory=dict
    )
    initial_positions: dict[TradingVenue | str, dict[ProductName, Decimal]] = Field(
        default_factory=dict
    )

    reuse_signals: bool = Field(
        default=True,
        description="Reuse pre-computed signals from child components in event-driven backtests.",
    )
    cache_features_df: bool = Field(
        default=True,
        description="""
            If True, keep features_df in memory after it's built.
            If False, recompute on every access. Trades memory for speed.
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
