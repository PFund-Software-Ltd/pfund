from __future__ import annotations

from typing import TYPE_CHECKING, Any, override

if TYPE_CHECKING:
    from collections.abc import Callable

import polars as pl

from pfund.backtest._mixins import portfolio_backtest_mixin, product_backtest_mixin
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

    def _validate_vectorized_backtest_inputs(self, df: pl.DataFrame) -> None:
        """Validates the strategy's backtest() and its input df before running a vectorized backtest.

        backtest() must exist and be callable with exactly one argument (df).
        Its return value can only be checked after the call, see _postprocess_for_vectorized_backtest().

        The df must be in LONG form: one row per (date, product, resolution) bar
        with shared OHLC columns, unique keys, sorted by date:

            date | product | resolution | open | high | low | close | ...
            d1   | BTC     | 1d         | ...
            d1   | ETH     | 1d         | ...
            d2   | BTC     | 1d         | ...

        Wide form (e.g. 'BTC_close', 'ETH_close' columns) is not supported.
        """
        import inspect

        from pfund.components.bar_dataframe import KEY_COLS

        backtest = getattr(self, "backtest", None)
        if not callable(backtest):
            raise NotImplementedError(
                f"{self} has no backtest() method, which VECTORIZED backtesting requires. "
                + "Implement `def backtest(self, df: pl.DataFrame) -> pl.DataFrame` "
                + "that returns the backtested df, or backtest it in EVENT_DRIVEN mode instead."
            )
        # dry-runs the argument matching of backtest(df) without calling it
        try:
            inspect.signature(backtest).bind(None)
        except TypeError as err:
            raise TypeError(
                f"backtest() must be callable with exactly 1 argument (df): {err}"
            ) from None

        # NOTE: volume is not required, the kernels ignore it (fills never depend on volume)
        ohlc_cols = ["open", "high", "low", "close"]
        missing_ohlc_cols = [c for c in ohlc_cols if c not in df.columns]
        if missing_ohlc_cols:
            raise ValueError(
                f"missing column(s) {missing_ohlc_cols} — the dataframe must be in LONG form: "
                + "one row per (date, product, resolution) bar with shared OHLC columns "
                + f"({', '.join(ohlc_cols)}). "
                + "Wide form (e.g. 'BTC_close', 'ETH_close') is not supported."
            )
        missing_key_cols = [c for c in KEY_COLS if c not in df.columns]
        if missing_key_cols:
            raise ValueError(
                f"backtesting requires columns ({', '.join(KEY_COLS)}) "
                + f"to identify each bar — missing column(s) {missing_key_cols}"
            )
        if df.is_empty():
            raise ValueError("the dataframe is empty, there is no data to backtest")
        null_key_cols = [c for c in KEY_COLS if df.get_column(c).has_nulls()]
        if null_key_cols:
            raise ValueError(
                f"key column(s) {null_key_cols} contain nulls — every bar must have a "
                + f"({', '.join(KEY_COLS)})"
            )
        if df.select(KEY_COLS).is_duplicated().any():
            raise ValueError(
                f"({', '.join(KEY_COLS)}) keys must be unique for backtesting"
            )
        # a string 'date' column sorts lexicographically, e.g. '2024-1-10' < '2024-1-9'
        date_dtype = df.schema["date"]
        if not date_dtype.is_temporal():
            raise ValueError(
                f"'date' column must be a date/datetime type, got {date_dtype}"
            )
        # bars are processed positionally per (product, resolution) combo: an
        # unsorted df would silently produce garbage results
        if not df.get_column("date").is_sorted():
            raise ValueError("the dataframe must be sorted by 'date' (ascending)")

        from pfund.engines.backtest_engine import BacktestEngine

        # the mode (product/portfolio) is unknown until backtest() runs, so both modes'
        # output columns are reserved, plus the engine's chunk column;
        # an input column with the same name would be silently overwritten
        reserved_cols = {
            *product_backtest_mixin.OUTPUT_COLS,
            *portfolio_backtest_mixin.OUTPUT_COLS,
            BacktestEngine.CHUNK_COL,
        }
        clashing_cols = [c for c in df.columns if c in reserved_cols]
        if clashing_cols:
            raise ValueError(
                f"input column(s) {clashing_cols} clash with backtest output columns "
                + f"{sorted(reserved_cols)}, please rename them"
            )
        # '_'-prefixed columns are internal, dropped in _postprocess_for_vectorized_backtest()
        private_cols = [c for c in df.columns if c.startswith("_")]
        if private_cols:
            raise ValueError(
                f"input column(s) {private_cols} start with '_', which is reserved for "
                + "internal backtest columns (they are dropped from the result), please rename them"
            )

    def _preprocess_for_vectorized_backtest(self, df: pl.DataFrame) -> None:
        """Prepares a fresh vectorized backtest session for the df's class.

        Validates the inputs first, see _validate_vectorized_backtest_inputs().
        Then clears the product and portfolio registries of any previous configuration,
        then attaches the two ENTRY methods to the df's class so they survive native
        operations (filter/join/... construct plain dataframes of the same class):
            create_signal — product backtesting
            create_weight — portfolio backtesting

        The rest of each mode's methods are deliberately NOT attached here:
        backtest()'s name is shared by both modes, so the entry method attaches
        its mode's implementation when called — create_signal() attaches
        open_position/close_position and ProductBacktestMixin's backtest,
        create_weight() attaches PortfolioBacktestMixin's backtest. (Consequence:
        chains must run sequentially — finish one chain before starting one of
        the other mode.)
        """
        from pfund.backtest._mixins.portfolio_backtest_mixin import (
            PortfolioBacktestMixin,
        )
        from pfund.backtest._mixins.product_backtest_mixin import ProductBacktestMixin

        self._validate_vectorized_backtest_inputs(df)

        product_backtest_mixin._clear_registry()
        portfolio_backtest_mixin._clear_registry()

        df_class = type(df)
        df_class.create_signal = ProductBacktestMixin.create_signal  # ty: ignore[missing-attribute]
        df_class.create_weight = PortfolioBacktestMixin.create_weight  # ty: ignore[missing-attribute]

    def _postprocess_for_vectorized_backtest(self, backtest_df: Any) -> pl.DataFrame:
        """Validates the backtest DataFrame returned by backtest(), then merges internal
        debug columns into a single 'remark' column and drops them.

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
        # a lazy chain is fine, the backtest only needs its result
        if isinstance(backtest_df, pl.LazyFrame):
            backtest_df = backtest_df.collect()
        if not isinstance(backtest_df, pl.DataFrame):
            raise TypeError(
                f"{self.name}.backtest() must return the backtested polars DataFrame (or LazyFrame), got {type(backtest_df).__name__}. Did you forget to return df?"
            )
        is_backtest_never_called = bool(
            product_backtest_mixin._registry or portfolio_backtest_mixin._registry
        )
        if is_backtest_never_called:
            raise ValueError(
                f"{self.name}.backtest() configured create_signal()/create_weight() "
                + "but never called df.backtest(), "
                + "call it at the end of the chain and return its result"
            )
        # the result must come from df.backtest(), i.e. have one mode's full output
        cols = set(backtest_df.columns)
        if not (
            cols.issuperset(product_backtest_mixin.OUTPUT_COLS)
            or cols.issuperset(portfolio_backtest_mixin.OUTPUT_COLS)
        ):
            raise ValueError(
                f"{self.name}.backtest() must return the df from df.backtest(), "
                + "the result is missing the backtest output columns, "
                + f"expected {list(product_backtest_mixin.OUTPUT_COLS)} "
                + f"or {list(portfolio_backtest_mixin.OUTPUT_COLS)}"
            )

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
