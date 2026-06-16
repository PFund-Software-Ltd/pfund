from decimal import Decimal

from pydantic import Field, field_validator, PrivateAttr

from pfund.engines.settings.base_engine_settings import BaseEngineSettings
from pfund.enums import TradingVenue, BacktestMode
from pfund.typing import Currency, ProductName


class BacktestEngineSettings(BaseEngineSettings):
    # If not provided, will use the default initial balances in SimulatedBroker
    initial_balances: dict[TradingVenue | str, dict[Currency, Decimal]] = Field(
        default_factory=dict
    )
    initial_positions: dict[TradingVenue | str, dict[ProductName, Decimal]] = Field(
        default_factory=dict
    )

    _backtest_mode: BacktestMode = PrivateAttr(init=False)
    reuse_signals: bool = Field(
        default=False,
        description="""
        if True, reuses signals from dumped signal_df in _next() instead of recalculating the signals.
        This will make event-driven backtesting a LOT faster but inconsistent with live trading.
        """,
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

    @property
    def backtest_mode(self) -> BacktestMode:
        return self._backtest_mode

    @backtest_mode.setter
    def backtest_mode(self, mode: BacktestMode) -> None:
        from pfund_kit.style import RichColor, TextStyle, cprint

        self._backtest_mode = BacktestMode[mode.upper()]
        if self._backtest_mode == BacktestMode.EXACT and self.reuse_signals:
            cprint(
                "Warning: Reusing pre-computed signals to speed up event-driven backtesting,\n"
                + "i.e. computing signals on the fly will be skipped",
                style=TextStyle.BOLD + RichColor.YELLOW,
            )
