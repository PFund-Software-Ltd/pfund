# pyright: reportUninitializedInstanceVariable=false, reportUnknownMemberType=false, reportAttributeAccessIssue=false, reportUnknownVariableType=false, reportArgumentType=false, reportUnusedParameter=false, reportUnknownParameterType=false, reportUnknownArgumentType=false
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
    from pfund.datas.data_bar import BarData
    from pfund.datas.stores.market_data_store import BarUpdate
    from pfund.engines.contexts.backtest_engine_context import BacktestEngineContext
    from pfund.engines.settings.backtest_engine_settings import BacktestEngineSettings
    from pfund.entities.products.product_base import BaseProduct
    from pfund.typing import ComponentT, Signals
    from pfund._backtest.cv.base import CrossValidatorDatasetPeriods
    from pfund._backtest.dataset_splitter import DatasetPeriods

from pathlib import Path

import numpy as np
import narwhals as nw
from narwhals.dependencies import is_into_dataframe
from pfund_kit.style import cprint, RichColor, TextStyle

from pfund.enums import BacktestMode


class BacktestMixin:
    df: IntoDataFrame

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
        from pfund.datas.stores.market_data_store import MarketDataStore

        key_cols = [MarketDataStore.INDEX_COL, *MarketDataStore.PIVOT_COLS]
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

    def backtest(self, df: IntoDataFrame) -> IntoDataFrame:
        if self.backtest_mode == BacktestMode.VECTORIZED:
            if hasattr(super(), "backtest"):
                if not self.is_strategy():
                    raise TypeError(
                        f"{self.name} should not override backtest() method"
                    )

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
                if self.is_strategy():
                    raise NotImplementedError(
                        f"{self.name} has no backtest() method, please implement it or switch to EVENT_DRIVEN mode"
                    )

                # signalize the full features df in vectorized backtesting
                X = self.X
                signals = cast("Signals", self.signalize(X))
                signals_df = nw.DataFrame.from_dict(
                    data=signals,
                    backend=nw.get_native_namespace(X),
                )
                trading_df = self.get_df(
                    kind="trading", window_size=None, to_native=False
                )
                self.store._df = nw.concat([trading_df, signals_df], how="horizontal")
                self.store.persist_to_lakehouse()
                return self.df
        elif self.backtest_mode == BacktestMode.EVENT_DRIVEN:
            return self._backtest_loop(df)
        else:
            raise ValueError(f"Unknown backtest mode: {self.backtest_mode}")

    # EXTEND: support non-bar data types:
    # brainstorm: heapq.merge(market_data_df, news_data_df), with a dispatcher to create data updates for each data type
    def _backtest_loop(self, df: IntoDataFrame) -> IntoDataFrame:
        from pfund_kit.style import RichColor
        from pfund_kit.utils.progress_bar import track

        # OPTIMIZE: critical loop
        for row in track(
            df.iter_rows(named=False),
            total=df.shape[0],
            description=description,
            bar_style=RichColor.BRIGHT_YELLOW,
        ):
            ts, resolution, product_name, source_type, o, h, l, c, v = row  # pyright: ignore[reportGeneralTypeIssues, reportUnusedVariable]  # noqa: E741
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
            # TODO: convert update to StreamingMessage
            self.databoy._collect(...)

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

    def _is_signals_precomputed(self) -> bool:
        return (self.backtest_mode == BacktestMode.VECTORIZED) or (
            self.backtest_mode == BacktestMode.EVENT_DRIVEN
            and self.settings.reuse_signals
        )

    def _materialize(self):
        for data_store in self.data_stores.values():
            data_store.materialize()
        if self._is_signals_precomputed():
            self.store.materialize()

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
    def _get_supported_resolutions(  # pyright: ignore[reportGeneralTypeIssues]
        self, product: BaseProduct
    ) -> dict[Timeframe, list[int]]:
        """Gets supported resolutions for the product based on the trading venue.
        Overrides it in backtesting, supports only the primary resolution.
        """
        return {self.resolution.timeframe: [self.resolution.period]}
