from __future__ import annotations

from typing import TYPE_CHECKING, override

if TYPE_CHECKING:
    from collections.abc import Callable

import polars as pl

from pfund.backtest._mixins.backtest_mixin import BacktestMixin


class BacktestStrategyMixin(BacktestMixin):
    """Backtest behavior specific to strategies.

    Adopted onto a strategy via ``BacktestStrategyMixin._adopt(strategy)``.
    """
    # defined by the user's strategy
    backtest: Callable[[pl.DataFrame], pl.DataFrame]

    @override
    def _set_name(self, name: str):
        from pfund.components.strategies._dummy_strategy import _DummyStrategy

        # NOTE: isinstance, not type(self) is _DummyStrategy, since _adopt swaps in a Backtest subclass
        if name == _DummyStrategy.__name__ and not isinstance(self, _DummyStrategy):
            raise ValueError(
                f'strategy name "{_DummyStrategy.__name__}" is reserved, please use another name'
            )
        super()._set_name(name)

    @staticmethod
    def _postprocess_backtest_df(backtest_df: pl.DataFrame) -> pl.DataFrame:
        """Postprocesses backtest DataFrame by merging internal debug columns into a
        single 'remark' column and dropping them.

        Remark abbreviations:
            sc  = signal_change — signal direction flipped (buy <-> sell)
            pc  = position_change — position closed or flipped (new trade streak)
            ft  = first_trade — first trade in a filtered trade streak
            sl  = stop_loss — position closed by stop loss
            tp  = take_profit — position closed by take profit
            tw  = time_window — position closed by time window expiry

        Columns dropped without remark (pure intermediates):
            _agg_costs, _immediate_stop, _gap_through_stop,
            _stop_side, _trade_side, _first_stop_order
        """
        REMARK_COLS: dict[str, str] = {
            "_signal_change": "sc",
            "_position_change": "pc",
            "_first_trade": "ft",
            "_stop_loss": "sl",
            "_take_profit": "tp",
            "_time_window": "tw",
        }

        # each flag that is set contributes its abbreviation, unset ones are null and skipped,
        # e.g. "sc,tp"; a row with no flags gets ""
        remark_parts = [
            pl.when(pl.col(col)).then(pl.lit(abbr))
            for col, abbr in REMARK_COLS.items()
            if col in backtest_df.columns
        ]
        remark = (
            pl.concat_str(remark_parts, separator=",", ignore_nulls=True)
            if remark_parts
            else pl.lit("")
        )
        df = backtest_df.with_columns(remark.alias("remark"))

        # Drop all internal columns (prefixed with '_')
        return df.drop(col for col in df.columns if col.startswith("_"))

    # TODO: update order manager directly
    async def place_orders(self):
        pass
