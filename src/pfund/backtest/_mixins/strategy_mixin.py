from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, cast, override

if TYPE_CHECKING:
    from collections.abc import Generator

    from pfund_blocks.datas import MarketData

    from pfund.typing import Component

    type ChunkNum = int

from contextlib import contextmanager

import polars as pl

from pfund.backtest._mixins import portfolio_backtest_mixin, product_backtest_mixin
from pfund.backtest._mixins.backtest_mixin import BacktestMixin
from pfund.enums import BacktestMode


def _split_df_into_time_chunks(
    df: pl.DataFrame,
    num_chunks: int,
) -> list[tuple[pl.DataFrame, ChunkNum]]:
    """Splits a date-sorted df into contiguous, timestamp-safe chunks.

    The unique dates are split like np.array_split (the extra dates go to the first chunks),
    so the rows of one date never end up in two chunks, e.g. in long form.
    """
    if num_chunks < 1:
        raise ValueError("num_chunks must be greater than 0")
    if "date" not in df.columns:
        raise ValueError("chunked backtesting requires a 'date' column")

    date = df.get_column("date")
    if date.is_empty():
        raise ValueError("chunked backtesting requires at least one row")
    if date.has_nulls():
        raise ValueError("chunked backtesting does not support null dates")
    if not date.is_sorted():
        raise ValueError("the dataframe must be sorted by 'date' before chunking")

    # 0, 0, 1, 1, 2, ...: the index of each row's date among the unique dates
    time_ids = date.rle_id()
    num_times = cast("int", time_ids[-1]) + 1
    if num_chunks > num_times:
        raise ValueError(
            f"num_chunks ({num_chunks}) cannot exceed the number of unique "
            + f"timestamps ({num_times})"
        )

    # chunk k covers the dates [time_bounds[k], time_bounds[k + 1])
    q, r = divmod(num_times, num_chunks)
    time_bounds = [k * q + min(k, r) for k in range(num_chunks + 1)]
    # the first row of each bound's date, the last bound (num_times) maps to len(df)
    row_bounds = time_ids.search_sorted(
        pl.Series(time_bounds, dtype=time_ids.dtype), side="left"
    ).to_list()
    return [
        (df.slice(row_bounds[k], row_bounds[k + 1] - row_bounds[k]), k)
        for k in range(num_chunks)
    ]


@contextmanager
def _without_dfs(component: Component) -> Generator[None]:
    """Temporarily detaches the dfs of the component tree, i.e. its data stores' dfs and store._df, children included.

    Used around ray.put() in chunked backtesting: otherwise every Ray worker in flight holds a copy of the
    full dataset (~67MB per product-year of 1m bars, ~4GB at 1s), though a chunk task only uses its df_chunk.
    So inside a chunk task, the backtestee's own dfs are None,
    i.e. a user's backtest() must use the df passed in, not self.data_df/self.df.
    The dfs are restored on exit.
    """
    detached: list[tuple[Any, pl.DataFrame | None]] = []

    def _detach(comp: Component) -> None:
        for store in [comp.store, *comp.data_stores.values()]:
            detached.append((store, store._df))
            store._df = None
        for child in comp.components:
            # components always run locally in backtesting, never as ActorProxy
            _detach(cast("Component", child))

    try:
        _detach(component)
        yield
    finally:
        for store, df in detached:
            store._df = df


class BacktestStrategyMixin(BacktestMixin):
    """Backtest behavior specific to strategies.

    Adopted onto a strategy via ``BacktestStrategyMixin._adopt(strategy)``.
    """

    # column telling apart the standalone backtests (chunks) in a strategy's backtest df, 0 for an unchunked run
    CHUNK_COL: ClassVar[str] = "chunk_num"

    def backtest(
        self,
        X: pl.DataFrame,
        num_chunks: int = 1,
        num_cpus: int | None = None,
    ) -> pl.DataFrame:
        """Runs a vectorized backtest of the strategy on ``X``.

        Overrides the user's backtest(df): _adopt() puts this mixin before the user's
        strategy class in the MRO, so ``strategy.backtest(X)`` lands here first,
        then calls the user's backtest(df) on each chunk, see _backtest_chunk().

        Args:
            X: the df passed to the user's backtest(df) as is, usually self.X after engine.use_split().
                It has the market data columns (e.g. OHLCV), since add_data() always sets
                as_features=True in vectorized backtesting.
            num_chunks:
                Number of standalone backtests to split ``df`` into.
                if <= 1, process the whole df all at once (values below 1 are treated as 1).
                if > 1, use Ray for parallel processing.
                When chunked, the strategy's dfs (e.g. self.data_df, self.df) are None inside its backtest(),
                use the df passed in.
            num_cpus:
                Maximum number of chunks (Ray tasks) in flight at once, i.e. how many chunks run in parallel.
                if None, defaults to os.cpu_count().
                This will be ignored if Ray is not used (i.e. num_chunks = 1).

        Returns:
            The backtest df, all chunks concatenated in chunk order, with each row's chunk in the CHUNK_COL column.
            Use df.partition_by(CHUNK_COL, as_dict=True) to get the per-chunk dfs.
        """
        import os

        if num_chunks <= 1:
            return self._backtest_chunk(X)

        from itertools import islice

        import ray
        from pfund_kit.utils.progress_bar import ProgressBar
        from pfund_kit.utils.ray import (
            ray_logging_context,
            setup_logger_in_ray_task,
            setup_ray,
        )
        from ray.util.queue import Queue

        num_cpus = (os.cpu_count() or 1) if num_cpus is None else max(1, num_cpus)

        @ray.remote
        def ray_task(
            log_queue: Queue,
            # passed in as an ObjectRef, Ray resolves top-level ObjectRef args before the task runs
            strategy: BacktestStrategyMixin,
            df_chunk: pl.DataFrame,
            chunk_num: int,
        ) -> pl.DataFrame:
            logger = setup_logger_in_ray_task(strategy.logger.name, log_queue)
            try:
                return strategy._backtest_chunk(df_chunk, chunk_num=chunk_num)
            except Exception:
                # log it to the log file, then re-raise: ray.get() raises it in the driver,
                # which cancels the other chunks, so a partial set of chunks is never returned
                logger.exception(f"Error in Backtest-Chunk{chunk_num}:")
                raise

        df_chunks: list[tuple[pl.DataFrame, int]] = _split_df_into_time_chunks(
            X, num_chunks
        )

        self.logger.debug("setting up ray...")
        setup_ray()
        # serialize the strategy into the object store once, instead of once per chunk,
        # each task still deserializes its own copy, so chunks don't share state.
        # its dfs are left out, see _without_dfs()
        with _without_dfs(cast("Any", self)):
            strategy_ref: ray.ObjectRef[BacktestStrategyMixin] = ray.put(self)

        logger = self.logger
        with ray_logging_context(logger) as log_queue:
            unfinished: list[ray.ObjectRef] = []
            try:
                pending = iter(df_chunks)
                # chunks finish in any order, keep the results in chunk order,
                # chunks are contiguous date ranges, so the combined df stays sorted by (chunk_num, date)
                ref_to_chunk_num: dict[ray.ObjectRef, int] = {}
                # placeholders, every slot is filled before the chunks are combined
                results: list[pl.DataFrame] = cast(
                    "list[pl.DataFrame]", [None] * len(df_chunks)
                )

                def submit_next(
                    df_chunk: pl.DataFrame, chunk_num: int
                ) -> ray.ObjectRef:
                    # Ray's .remote() only takes positional arguments (for type checkers)
                    ref = ray_task.remote(log_queue, strategy_ref, df_chunk, chunk_num)
                    ref_to_chunk_num[ref] = chunk_num
                    return ref

                # sliding window: keep num_cpus chunks in flight, start the next chunk
                # as soon as one finishes, so a slow chunk doesn't leave the other workers idle
                unfinished = [submit_next(*item) for item in islice(pending, num_cpus)]
                with ProgressBar(
                    total=len(df_chunks),
                    description=f"Backtesting {self.name} ({len(df_chunks)} chunks, {min(num_cpus, len(df_chunks))} in parallel)",
                ) as pbar:
                    while unfinished:
                        finished, unfinished = ray.wait(unfinished, num_returns=1)
                        results[ref_to_chunk_num[finished[0]]] = ray.get(finished[0])
                        pbar.advance(1)
                        if (item := next(pending, None)) is not None:
                            unfinished.append(submit_next(*item))
            except BaseException:
                # e.g. KeyboardInterrupt or a crashed worker: cancel the in-flight chunks,
                # otherwise they keep running after the backtest returns, and re-raise.
                # force=True kills the worker, a non-forced cancel doesn't stop a task that's already running
                for ref in unfinished:
                    ray.cancel(ref, force=True)
                raise

        # combine the chunks into one df, ordered by (chunk_num, date): chunks are contiguous date ranges,
        # and the results are kept in chunk order.
        return pl.concat(results, how="vertical")

    @override
    def add_data(
        self, *args: Any, as_features: bool = True, **kwargs: Any
    ) -> list[MarketData]:
        """Same as ComponentMixin.add_data(), but always sets as_features=True in vectorized backtesting.

        The vectorized backtest kernel needs the market data (OHLC) in the df, and X is that df,
        see backtest(). It doesn't affect event-driven backtesting, where trade(X) gets the
        latest bar via self.get_data(), so X doesn't need the market data.
        """
        if self.backtest_mode == BacktestMode.VECTORIZED and not as_features:
            self.logger.warning(
                f"{self.name}: as_features set to True in vectorized backtesting, "
                + "i.e. X (or features_df) will have the OHLC columns."
            )
            as_features = True
        return super().add_data(*args, as_features=as_features, **kwargs)

    add_market_data = add_data

    def _backtest_chunk(self, df: pl.DataFrame, chunk_num: int = 0) -> pl.DataFrame:
        """Runs the user's backtest(df) on one chunk, in between the pre/postprocessing.

        Chunks are standalone backtests, the CHUNK_COL column tells them apart;
        an unchunked run is chunk 0, so every strategy backtest df has the same schema.
        """
        self._preprocess_for_vectorized_backtest(df)
        # the user's backtest(), next in the MRO after this mixin
        backtest_df = cast("Any", super()).backtest(df)
        backtest_df = self._postprocess_for_vectorized_backtest(backtest_df)
        return backtest_df.with_columns(
            pl.lit(chunk_num).alias(BacktestStrategyMixin.CHUNK_COL)
        )

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

        # the user's backtest(), self.backtest is this mixin's override
        backtest = getattr(super(), "backtest", None)
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

        # the mode (product/portfolio) is unknown until backtest() runs, so both modes'
        # output columns are reserved, plus the engine's chunk column;
        # an input column with the same name would be silently overwritten
        reserved_cols = {
            *product_backtest_mixin.OUTPUT_COLS,
            *portfolio_backtest_mixin.OUTPUT_COLS,
            BacktestStrategyMixin.CHUNK_COL,
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
        """Validates the backtest DataFrame returned by backtest(), adds a 'remark'
        debug column, then drops internal ('_'-prefixed) columns.

        The remark is derived from the product backtest's output columns only, per
        (product, resolution), e.g. "pc,sl"; a row with no event gets "".
        Portfolio backtesting has no such events, its remark is always "".

        Remark abbreviations, and the row each one is on:
            sc  = signal_change — the signal starts a new streak (first signal or buy <-> sell), signal row
            pc  = position_change — the position opens, closes or flips, fill row
            sl  = stop_loss — a stop below avg_price (long) / above it (short) fires next bar,
                  order row (same row as stop_price)
            tp  = take_profit — a stop on the profit side of avg_price fires next bar, order row
            ex  = exit — a close with no order and no stop on the row before,
                  i.e. a time_window close or an out-of-universe close, fill row

        Limitations (the outputs don't carry the reason):
            a trailing stop reads as sl, or as tp once it trails above avg_price (long) / below it (short);
            ex doesn't tell time_window from out-of-universe;
            a time_window close that beats a pending order on the same bar gets no ex.
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

        remark = (
            self._build_remark_col_for_product_kernel()
            if cols.issuperset(product_backtest_mixin.OUTPUT_COLS)
            else pl.lit("")
        )
        df = backtest_df.with_columns(remark.alias("remark"))

        # Drop all internal columns (prefixed with '_')
        return df.drop(col for col in df.columns if col.startswith("_"))

    @staticmethod
    def _build_remark_col_for_product_kernel() -> pl.Expr:
        """The 'remark' expression of a product backtest df, see _postprocess_for_vectorized_backtest()."""
        from pfund.components.bar_dataframe import PIVOT_COLS

        # the outputs are nan (not null) where nothing happened, nan -> null for null-aware ops
        def output(col: str) -> pl.Expr:
            return pl.col(col).fill_nan(None)

        signal_ffill = output("signal").forward_fill()
        position_sign = output("position").sign()
        has_stop = output("stop_price").is_not_null()
        # > 0: the stop is on the losing side of avg_price
        stop_vs_avg = output("position") * (output("avg_price") - output("stop_price"))
        is_exit_without_order = (
            output("trade_size").is_not_null()
            & output("order_size").shift(1).is_null()
            & output("stop_price").shift(1).is_null()
        )
        events: dict[str, pl.Expr] = {
            "sc": signal_ffill.is_not_null()
            & (signal_ffill != signal_ffill.shift(1)).fill_null(True),
            "pc": position_sign.is_not_null()
            & (position_sign != position_sign.shift(1).fill_null(0)),
            "sl": has_stop & (stop_vs_avg > 0),
            "tp": has_stop & (stop_vs_avg < 0),
            "ex": is_exit_without_order,
        }
        # each event that is set contributes its abbreviation, unset ones are null and skipped
        return pl.concat_str(
            [
                pl.when(event.over(PIVOT_COLS)).then(pl.lit(abbr))
                for abbr, event in events.items()
            ],
            separator=",",
            ignore_nulls=True,
        )

    # TODO: update order manager directly
    async def place_orders(self):
        pass
