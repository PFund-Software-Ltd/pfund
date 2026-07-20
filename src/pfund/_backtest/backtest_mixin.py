# pyright: reportUninitializedInstanceVariable=false, reportUnknownMemberType=false, reportAttributeAccessIssue=false, reportUnknownVariableType=false, reportArgumentType=false, reportUnusedParameter=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportGeneralTypeIssues=false
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, cast

from typing_extensions import override

if TYPE_CHECKING:
    from typing import TypeVar

    from numpy.typing import NDArray
    from narwhals.typing import IntoDataFrame, IntoSeries
    from pfeed.storages.storage_config import StorageConfig

    # constrained (not bound) so the exact input type is preserved on the way out:
    # DataFrame in -> DataFrame out, Series in -> Series out, ndarray in -> ndarray out
    DataT = TypeVar("DataT", IntoDataFrame, IntoSeries, "NDArray[Any]")

    from pfund.datas.timeframe import Timeframe
    from pfund.datas.data_base import BaseData
    from pfund.datas.data_bar import BarData
    from pfund.datas.stores.market_data_store import BarUpdate
    from pfund.engines.contexts.backtest_engine_context import BacktestEngineContext
    from pfund.engines.settings.backtest_engine_settings import BacktestEngineSettings
    from pfund.entities.products.product_base import BaseProduct
    from pfund.typing import ComponentT, Signals, Component, ComponentName
    from pfund._backtest.cv.base import CrossValidatorDatasetPeriods
    from pfund._backtest.dataset_splitter import DatasetPeriods

from pathlib import Path

import numpy as np
import narwhals as nw
from narwhals.dependencies import is_into_dataframe
from pfund_kit.style import cprint, RichColor, TextStyle
from pfeed.enums import DataCategory

from pfund.components.bar_dataframe import (
    INDEX_COL as BAR_INDEX_COL,
    KEY_COLS as BAR_KEY_COLS,
    PIVOT_COLS as BAR_PIVOT_COLS,
    align_df_to_spine,
    pivot_long_to_wide,
    reorder_key_cols,
    validate_spine_df,
)
from pfund.enums import BacktestMode


class BacktestMixin:
    df: IntoDataFrame

    def __mixin_post_init__(self, *args: Any, **kwargs: Any):
        super().__mixin_post_init__(*args, **kwargs)
        # top components are the ones added by engine.add_strategy/model/feature()
        self._is_top_component = False

    @property
    def context(self) -> BacktestEngineContext:
        assert self._context is not None, "context is not set"
        return cast("BacktestEngineContext", self._context)

    @property
    def settings(self) -> BacktestEngineSettings:
        return self.context.settings

    @property
    def backtest_mode(self) -> BacktestMode:
        return self._context.mode

    @property
    def dataset_periods(self) -> DatasetPeriods | list[CrossValidatorDatasetPeriods]:
        return self.context.dataset_periods

    @staticmethod
    def _validate_backtest_signature(
        func: Callable[[IntoDataFrame], IntoDataFrame],
    ):
        """Validates the signature of the backtest() function.
        The backtest() function must accept exactly 1 argument (df) and return the backtested dataframe.
        """
        import ast
        import inspect
        import textwrap

        sig = inspect.signature(func)
        params = [p for p in sig.parameters if p not in ("self", "cls")]
        if len(params) != 1:
            raise TypeError(
                f"backtest() must accept exactly 1 argument (df), got {len(params)}: {params}"
            )

        try:
            source = textwrap.dedent(inspect.getsource(func))
            tree = ast.parse(source)
            func_def = tree.body[0]
            has_return = any(
                isinstance(node, ast.Return) and node.value is not None
                for node in ast.walk(func_def)
            )
            if not has_return:
                raise TypeError(
                    "backtest() must return the backtested dataframe. No return statement found. Did you forget to return df?"
                )
        except OSError:
            pass  # source not available (e.g. built-in), skip check

    @staticmethod
    def _setup_backtest_df_for_vectorized_mode(df: IntoDataFrame) -> IntoDataFrame:
        """Prepare a native dataframe (pandas/polars) for FAST backtesting.

        Attaches the two ENTRY methods to the df's class so they survive native
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

        Validates that the df is in LONG form — one row per bar with shared OHLCV
        columns; for multiple (product, resolution) combos, one row per
        (date, resolution, product):

            date | resolution | product | open | high | low | close | volume
            d1   | 1d         | BTC     | ...
            d1   | 1d         | ETH     | ...
            d2   | 1d         | BTC     | ...

        Wide form (e.g. 'BTC_close', 'ETH_close' columns) is not supported.

        Product backtesting is configured per (product, resolution) combo and
        executed once — the configure chain only REGISTERS inputs:

            for product, resolution in combos:
                (df.filter(...)  # one combo
                   .create_signal(...).open_position(...).close_position(...))
            df = df.backtest()   # ONE call: kernel per configured combo

        Dynamic periods (optional): wrap the configure loop in an outer period
        loop and pass data_range=(period_start, period_end) to create_signal()
        to register each combo per period with a point-in-time universe — all
        params may vary per period. Registration controls instructions, never
        data: the kernel still runs once per combo over its FULL rows, so
        positions carry across periods and stops stay live on every bar (see
        create_signal's docstring).

        Every df must carry (date, resolution, product) — MarketDataStore
        guarantees them; a single-product df is simply one combo, so the fluent
        df.create_signal(...)....backtest() chain works on it directly.
        Calling _setup_backtest_df_for_vectorized_mode starts a fresh session (clears the product
        and portfolio registries of any previous configuration).

        Returns the df unchanged.
        """
        import narwhals as nw

        from pfund._backtest import portfolio_backtest_mixin, product_backtest_mixin
        from pfund._backtest.portfolio_backtest_mixin import PortfolioBacktestMixin
        from pfund._backtest.product_backtest_mixin import ProductBacktestMixin

        product_backtest_mixin._clear_registry()
        portfolio_backtest_mixin._clear_registry()

        df_class = type(df)
        df_class.create_signal = ProductBacktestMixin.create_signal
        df_class.create_weight = PortfolioBacktestMixin.create_weight

        nw_df = nw.from_native(df)
        if "close" not in nw_df.columns:
            raise ValueError(
                "'close' column not found — the dataframe must be in LONG form: "
                + "one row per bar with shared OHLCV columns "
                + "(one row per (date, product) for multiple products). "
                + "Wide form (e.g. 'BTC_close', 'ETH_close') is not supported."
            )
        from pfund.components.bar_dataframe import KEY_COLS

        key_cols = KEY_COLS
        missing = [c for c in key_cols if c not in nw_df.columns]
        if missing:
            raise ValueError(
                f"backtesting requires columns ({', '.join(key_cols)}) "
                + f"to identify each bar — missing column(s) {missing}"
            )
        if len(nw_df.select(*key_cols).unique()) != len(nw_df):
            raise ValueError(
                f"({', '.join(key_cols)}) keys must be unique for backtesting"
            )
        # bars are processed positionally per (product, resolution) combo: an
        # unsorted df would silently produce garbage results
        if not nw_df.get_column("date").is_sorted():
            raise ValueError("the dataframe must be sorted by 'date' (ascending)")
        return df

    @property
    def _source_artifact(self) -> Path:
        if self._source_artifact_path is not None:
            return self._source_artifact_path

        import inspect

        from pfund._backtest.backtest_mixin import BacktestMixin

        mro = type(self).__mro__
        source_class = mro[mro.index(BacktestMixin) + 1]
        source_file = inspect.getsourcefile(source_class)
        if source_file is None:
            raise ValueError(f"cannot locate source file for {source_class.__name__}")
        return Path(source_file)

    @property
    def _data_artifact(self) -> IntoDataFrame:
        if self.backtest_mode == BacktestMode.VECTORIZED:
            return self.store._df
        return super()._data_artifact

    def set_top_component(self):
        self._is_top_component = True

    def backtest(self, df: IntoDataFrame) -> IntoDataFrame:
        if self.backtest_mode == BacktestMode.VECTORIZED:
            if hasattr(super(), "backtest"):
                if not self.is_strategy():
                    raise TypeError(f"{self.name} should not have a backtest() method")

                super_backtest = cast(
                    "Callable[[IntoDataFrame], Any]", super().backtest
                )
                self._validate_backtest_signature(super_backtest)

                # only after we know a valid backtest() exists: prepares the df for the
                # user's chain (attaches create_signal/create_weight, validates LONG form,
                # clears registries); returns df unchanged
                df = self._setup_backtest_df_for_vectorized_mode(df)

                backtest_df: Any = super_backtest(df)

                # check if returns a df
                if not is_into_dataframe(backtest_df):
                    raise TypeError(
                        f"{self.name}.backtest() must return the backtested dataframe, got {type(backtest_df).__name__}. Did you forget to return df?"
                    )
                # check if returns a new df
                if backtest_df is df:
                    cprint(
                        f"WARNING: {self.name} backtest() returned the same df unchanged.\n"
                        + "This is fine if you only used native e.g. Polars/Pandas operations on the original df.\n"
                        + "However, [italic]this is an ERROR[/italic] if you called backtest methods such as "
                        + "create_signal(), open_position(), or close_position() —\n"
                        + "these return a new df, so you must reassign: df = df.create_signal(...) and return the new df",
                        style=TextStyle.BOLD + RichColor.RED,
                    )
                return cast("IntoDataFrame", backtest_df)
            else:
                self.store._df = self._vectorized_backtest(nw.from_native(df))
                self.store.persist_to_lakehouse()
                return self.df
        elif self.backtest_mode == BacktestMode.EVENT_DRIVEN:
            return self._event_driven_backtest(df)
        else:
            raise ValueError(f"Unknown backtest mode: {self.backtest_mode}")

    def _vectorized_backtest(self, X: nw.DataFrame[Any]) -> nw.DataFrame[Any]:
        """Runs a vectorized backtest and completes the feature-only trading dataframe with the resulting signals."""
        if self.is_strategy():
            raise NotImplementedError(
                f"{self.name} has no backtest() method, please implement it or switch to EVENT_DRIVEN mode"
            )
        signals = cast("Signals", self.signalize(X.to_native()))
        signals_df = nw.DataFrame.from_dict(
            data=signals,
            backend=nw.get_native_namespace(X),
        )
        return nw.concat([X, signals_df], how="horizontal")

    # EXTEND: support non-bar data types:
    # brainstorm: heapq.merge(market_data_df, news_data_df), with a dispatcher to create data updates for each data type
    # TODO: write trading_df (trading_df.delta) and compare it with the one generated by vectorized backtesting
    # TODO: need to build a clock? coz looping bars has time gaps
    def _event_driven_backtest(self, df: nw.DataFrame[Any]) -> nw.DataFrame[Any]:
        from pfund_kit.style import RichColor
        from pfund_kit.utils.progress_bar import track

        # OPTIMIZE: critical loop
        for row in track(
            df.iter_rows(named=False),
            total=df.shape[0],
            description=description,
            bar_style=RichColor.BRIGHT_YELLOW,
            disable=not self.context.pfund_config.show_progress_bar,
        ):
            ts, resolution, product_name, source_type, o, h, l, c, v = row  # pyright: ignore[reportUnusedVariable]  # noqa: E741
            data = cast("BarData", self.get_data(product_name, resolution))
            update: BarUpdate = {
                "ts": ts,
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": v,
                "is_incremental": False,
                "msg_ts": None,
                "extra": {},
            }
            self._databoy.collect(...)

        # TODO: return the backtested dataframe, how?
        return df

    def _get_dataset(
        self,
        key: Literal["train_set", "dev_set", "test_set"],
        data: IntoDataFrame | IntoSeries | NDArray[Any] | None = None,
        fold: int | None = None,
    ) -> IntoDataFrame | IntoSeries | NDArray[Any] | None:
        """Slice a dataset down to `key`'s date window.

        dataset_periods only computes the date boundaries; this turns those
        dates into rows. By default it slices the full feature dataset (self.df).
        Pass `data` to slice an external object (e.g. a target y) by the SAME
        window so it stays aligned with the features; `data` must be row-aligned
        1:1 with self.df (same length and order).

        Args:
            data: object to slice instead of self.df. A DataFrame/Series is
                filtered by the mask; a bare ndarray is sliced positionally by
                it. Leave None to slice self.df.
            fold: which CV fold to slice. Leave None for a ratio split; it is
                required (and picks the fold) under cross-validation.

        Returns None when the dataset isn't materialized yet, the segment
        collapsed to 0 days ((None, None)), or the split doesn't define `key`
        (CV folds have no test_set — the test hold-out lives outside the folds).
        """
        import narwhals as nw

        periods = self.dataset_periods
        if isinstance(periods, list):
            if fold is None:
                raise ValueError(
                    f"{self.name} uses cross-validation ({len(periods)} folds); "
                    + f"pass fold=0..{len(periods) - 1} to pick one"
                )
            if not 0 <= fold < len(periods):
                raise IndexError(f"fold {fold} out of range for {len(periods)} folds")
            period = periods[fold]
        else:
            if fold is not None:
                raise ValueError(
                    f"{self.name} uses a ratio split, not cross-validation; "
                    + "fold must be None"
                )
            period = periods

        window = period.get(key)
        if window is None:
            return None
        start, end = window
        if start is None or end is None:
            return None

        # build the row mask once from self.df's date column (the only source
        # guaranteed to carry `date`), then apply it to whatever we're slicing so
        # X and any aligned `data` (y) share the window
        date = nw.from_native(self.df)["date"].dt.date()
        mask = (date >= start) & (date <= end)
        if data is None:
            return nw.from_native(self.df).filter(mask).to_native()
        else:
            # `data` is sliced positionally, so it MUST be row-aligned 1:1 with self.df;
            # a length mismatch means the slice would silently misalign X and y
            n_ref = mask.len()
            n_data = (
                data.shape[0]
                if isinstance(data, np.ndarray)
                else len(nw.from_native(data, allow_series=True))
            )
            if n_data != n_ref:
                raise ValueError(
                    f"{self.name}: `data` must be row-aligned 1:1 with the dataset "
                    + f"({n_ref} rows), but got {n_data} rows"
                )

            if isinstance(data, np.ndarray):
                # bare ndarray has no date column -> positional slice by the same mask
                return data[mask.to_numpy()]
            return nw.from_native(data, allow_series=True).filter(mask).to_native()

    @property
    def train_set(self) -> IntoDataFrame | None:
        """Rows of the training split."""
        return self.get_train_set()

    training_set = train_set

    @property
    def dev_set(self) -> IntoDataFrame | None:
        """Rows of the dev (validation) split."""
        return self.get_dev_set()

    val_set = validation_set = development_set = dev_set

    @property
    def test_set(self) -> IntoDataFrame | None:
        """Rows of the test split."""
        return self.get_test_set()

    def get_train_set(self, fold: int | None = None) -> IntoDataFrame | None:
        """Rows of the training split.

        Args:
            fold: which cross-validation fold to slice, 0-indexed. Leave None
                for a ratio split (there is a single train set); required under
                cross-validation, where it selects the fold.
        """
        return cast("IntoDataFrame | None", self._get_dataset("train_set", fold=fold))

    get_training_set = get_train_set

    def get_dev_set(self, fold: int | None = None) -> IntoDataFrame | None:
        """Rows of the dev (validation) split.

        Args:
            fold: which cross-validation fold to slice, 0-indexed. Leave None
                for a ratio split (there is a single dev set); required under
                cross-validation, where it selects the fold.
        """
        return cast("IntoDataFrame | None", self._get_dataset("dev_set", fold=fold))

    get_val_set = get_validation_set = get_development_set = get_dev_set

    def get_test_set(self, fold: int | None = None) -> IntoDataFrame | None:
        """Rows of the test (hold-out) split.

        Args:
            fold: which cross-validation fold to slice, 0-indexed. Leave None
                for a ratio split. Note CV folds have no test set (the test
                hold-out lives outside the folds), so this returns None for
                any fold under cross-validation.
        """
        return cast("IntoDataFrame | None", self._get_dataset("test_set", fold=fold))

    def split_dataset(
        self,
        data: DataT,
        fold: int | None = None,
    ) -> tuple[DataT | None, DataT | None, DataT | None]:
        """Split `data` into (train, dev, test) slices using the same date
        windows as the features (self.df).

        `data` is sliced positionally by a mask built from self.df, so it needs
        no `date` column of its own — it only has to be row-aligned 1:1 with
        self.df (same length and order). A target y, class labels, sample
        weights, or any per-row companion of the features therefore split on the
        identical boundaries as train_set/dev_set/test_set. This is how X and y
        stay aligned: put each through the same window.

            close = next(c for c in model.df.columns if c.endswith(":close"))
            y = model.df.get_column(close).shift(-1) / model.df.get_column(close) - 1.0
            X_train, X_val, _ = model.split_dataset(model.df)
            y_train, y_val, _ = model.split_dataset(y)
            model.fit(X_train, y_train)

        Args:
            data: a DataFrame, Series, or ndarray row-aligned 1:1 with self.df.
            fold: which cross-validation fold to slice, 0-indexed. Leave None for
                a ratio split; required under cross-validation, where it selects
                the fold (test is always None per fold — the hold-out lives
                outside the folds).

        Returns a (train, dev, test) tuple; any segment that collapsed to 0 days
        or is undefined for the split is None.
        """
        return cast(
            "tuple[DataT | None, DataT | None, DataT | None]",
            (
                self._get_dataset("train_set", data=data, fold=fold),
                self._get_dataset("dev_set", data=data, fold=fold),
                self._get_dataset("test_set", data=data, fold=fold),
            ),
        )

    @override
    def _add_component(
        self,
        component: ComponentT,
        resolution: str,
        name: str,
        df_form: Literal["wide", "long"],
        storage_config: StorageConfig | None,
        # NOTE: non-backtesting kwargs are ignored, e.g. ray_actor_options, ray_kwargs, etc.
        **kwargs: Any,
    ) -> ComponentT:
        source_artifact_path = component._source_artifact_path
        Component = type(component)
        if component.is_strategy():
            from pfund.components.strategies.strategy_backtest import BacktestStrategy

            component = BacktestStrategy(
                Component,
                *component.__pfund_args__,
                **component.__pfund_kwargs__,
            )
        elif component.is_model():
            from pfund.components.models.model_backtest import BacktestModel

            component = BacktestModel(
                Component,
                component.model,
                *component.__pfund_args__,
                **component.__pfund_kwargs__,
            )
        elif component.is_feature():
            from pfund.components.features.feature_backtest import BacktestFeature

            Component = type(component)
            component = BacktestFeature(
                Component,
                *component.__pfund_args__,
                **component.__pfund_kwargs__,
            )
        else:
            raise ValueError(f"Unsupported component type: {component}")
        if source_artifact_path is not None:
            component._set_source_artifact_path(str(source_artifact_path))
        return super()._add_component(
            component=component,
            resolution=resolution,
            name=name or Component.__name__,
            df_form=df_form,
            storage_config=storage_config,
        )

    def _is_dummy_strategy(self) -> bool:
        from pfund.components.strategies._dummy_strategy import _DummyStrategy

        return isinstance(self, _DummyStrategy)

    @property
    def _reuse_child_component_signals(self):
        return not self._is_top_component and self.settings.reuse_signals

    @override
    def _get_components_signals(self, data: BarData) -> dict[ComponentName, Signals]:
        return {
            component.name: component.store.get_signals(data)
            for component in self.components
        }

    # data df has already been loaded in full in backtesting, so no need to update
    @override
    def _update_data_df(self, data: BaseData) -> None:
        pass

    @override
    def step(self, data: BarData):
        try:
            if self._reuse_child_component_signals:
                # NOTE: child component's signals are already available (loaded during materialization), reuse them
                self._latest_signals = self.store.get_signals(data)
                return
            return super().step(data)
        except Exception:
            self.logger.exception("Error forwarding data:")

    # no-op in backtesting, zeromq is not in use
    @override
    def _publish_signals(self, signals: Signals, data: BarData):
        pass

    @override
    def _materialize(self):
        for data_store in self.data_stores.values():
            data_store.materialize()
        if self.backtest_mode == BacktestMode.VECTORIZED:
            # child component will try to load its trading_df
            if not self._is_top_component:
                self.store.materialize()
            if self.store._df is None:
                X = self._merge_signals_dfs()
                if not self._is_top_component:
                    cprint(
                        f"No persisted trading dataframe found for '{self.name}'. "
                        + "Running its vectorized backtest on the fly...",
                        style=TextStyle.BOLD + RichColor.YELLOW,
                    )
                    self.store._df = self._vectorized_backtest(X)
                    self.store.persist_to_lakehouse()
                else:
                    self.store._df = X
        elif self.backtest_mode == BacktestMode.EVENT_DRIVEN:
            if self._reuse_child_component_signals:
                self.store.materialize()
        else:
            raise ValueError(f"Unknown backtest mode: {self.backtest_mode}")

    @override
    def _gather(self) -> None:
        if self._is_dummy_strategy():
            assert len(self.components) == 1, (
                "dummy strategy must have exactly one component"
            )
            component = self.components[0]
            component._gather()
            self._is_gathered = True
        else:
            return super()._gather()

    @override
    def _get_supported_resolutions(
        self, product: BaseProduct
    ) -> dict[Timeframe, list[int]]:
        """Gets supported resolutions for the product based on the trading venue.
        Overrides it in backtesting, supports only the primary resolution.
        """
        return {self.resolution.timeframe: [self.resolution.period]}

    def _merge_signals_dfs(self) -> nw.DataFrame[Any]:
        """Build this component's historical feature frame.

        The component's own data establishes its receiving bar spine when
        present. Otherwise, same-resolution child signals establish the spine.
        Each child component's signal dataframe is then aligned to that spine
        and merged as feature columns. Once established, child signals can add
        values but cannot add rows to the receiving universe.

        Strictly equal resolutions use exact alignment; other resolutions use
        close-time as-of alignment. Long-form child signals are aligned per
        product. Wide-form child signals are aligned as one cross-sectional
        date stream and are broadcast by date when the receiver is long-form.
        """

        def _align_wide_child_signals(
            base_df: nw.DataFrame[Any],
            child: Component,
            child_signals: nw.DataFrame[Any],
            *,
            alignment_mode: Literal["exact", "asof"],
        ) -> nw.DataFrame[Any]:
            # A wide dataframe is one cross-sectional stream keyed by date.
            # Add a synthetic product and the known component resolutions so
            # the canonical bar aligner can apply the same close-time rules.
            wide_stream = "__wide_stream__"
            alignment_spine = (
                base_df.select(BAR_INDEX_COL)
                .unique()
                .with_columns(
                    nw.lit(wide_stream).alias("product"),
                    nw.lit(str(self.resolution)).alias("resolution"),
                )
            )
            alignment_contributor = child_signals.with_columns(
                nw.lit(wide_stream).alias("product"),
                nw.lit(str(child.resolution)).alias("resolution"),
            )
            return align_df_to_spine(
                alignment_spine,
                alignment_contributor,
                mode=alignment_mode,
            ).drop(BAR_PIVOT_COLS)

        def _align_long_child_signals_to_wide_parent(
            base_df: nw.DataFrame[Any],
            child_signals: nw.DataFrame[Any],
            *,
            alignment_mode: Literal["exact", "asof"],
        ) -> nw.DataFrame[Any]:
            # A wide parent has no product rows on which to perform an as-of
            # join. Temporarily expand its receiving dates across the child's
            # products, align each product stream independently, then pivot the
            # aligned result back to the parent's wide form.
            child_products = child_signals.select("product").unique()
            alignment_spine = (
                base_df.select(BAR_INDEX_COL)
                .unique()
                .join(child_products, how="cross")
                .with_columns(nw.lit(str(self.resolution)).alias("resolution"))
                .select(BAR_KEY_COLS)
            )
            aligned_signals = align_df_to_spine(
                alignment_spine,
                child_signals,
                mode=alignment_mode,
            )
            return pivot_long_to_wide(
                aligned_signals,
                index_col=BAR_INDEX_COL,
                pivot_cols=BAR_PIVOT_COLS,
            )

        def _join_child_signals(
            base_df: nw.DataFrame[Any],
            child: Component,
            child_signals: nw.DataFrame[Any],
        ) -> nw.DataFrame[Any]:
            alignment_mode = (
                "exact" if child.resolution.is_strict_equal(self.resolution) else "asof"
            )
            if self.df_form == "long" and child.df_form == "long":
                join_cols = BAR_KEY_COLS
                child_signals = align_df_to_spine(
                    base_df,
                    child_signals,
                    mode=alignment_mode,
                )
            elif self.df_form == "wide" and child.df_form == "long":
                join_cols = [BAR_INDEX_COL]
                child_signals = _align_long_child_signals_to_wide_parent(
                    base_df,
                    child_signals,
                    alignment_mode=alignment_mode,
                )
            else:
                join_cols = [BAR_INDEX_COL]
                child_signals = _align_wide_child_signals(
                    base_df,
                    child,
                    child_signals,
                    alignment_mode=alignment_mode,
                )

            child_value_cols = [
                col for col in child_signals.columns if col not in join_cols
            ]
            conflicting_cols = [
                col for col in child_value_cols if col in base_df.columns
            ]
            if conflicting_cols:
                raise ValueError(
                    f"child signal columns already exist in {self.name}'s features: "
                    + f"{conflicting_cols}"
                )
            return base_df.join(
                child_signals,
                on=join_cols,
                how="left",
            ).sort(join_cols)

        children_signals = [
            (
                child,
                child.get_df(kind="signals", window_size=None, to_native=False),
            )
            for child in self.components
        ]
        data_dfs = {
            category: data_store.get_df(to_native=False)
            for category, data_store in self.data_stores.items()
            # NOTE: market data df is ALWAYS included to create the "spine" of the df
            if (category == DataCategory.MARKET_DATA or data_store.data_as_features)
            and data_store.get_datas()
        }
        if data_dfs:
            df = self._merge_data_dfs(data_dfs)
            market_data_store = self.market_data_store
            if not market_data_store.data_as_features:
                market_data_df = market_data_store.get_df(to_native=False)
                market_data_cols = [
                    col
                    for col in market_data_df.columns
                    if col not in BAR_KEY_COLS + market_data_store.METADATA_COLS
                ]
                df = df.drop(market_data_cols, strict=False)

            validate_spine_df(df)
            if self.df_form == "wide":
                value_cols = [col for col in df.columns if col not in BAR_KEY_COLS]
                if value_cols:
                    df = pivot_long_to_wide(
                        df,
                        index_col=BAR_INDEX_COL,
                        pivot_cols=BAR_PIVOT_COLS,
                    )
                else:
                    df = df.select(BAR_INDEX_COL).unique().sort(BAR_INDEX_COL)
        else:
            spine_cols = BAR_KEY_COLS if self.df_form == "long" else [BAR_INDEX_COL]
            spine_dfs = [
                child_signals.select(spine_cols)
                for child, child_signals in children_signals
                if child.resolution.is_strict_equal(self.resolution)
                and (self.df_form == "wide" or child.df_form == "long")
            ]
            if not spine_dfs:
                required_child = (
                    "same-resolution child signals"
                    if self.df_form == "wide"
                    else "same-resolution long-form child signals"
                )
                raise ValueError(
                    f"{self.name} has no data or {required_child} to establish "
                    + "its bar spine"
                )
            df = nw.concat(spine_dfs).unique().sort(spine_cols)

            if self.df_form == "long":
                validate_spine_df(df)

        for child, child_signals in children_signals:
            df = _join_child_signals(df, child, child_signals)

        return reorder_key_cols(df, df_form=self.df_form)
