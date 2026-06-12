# pyright: reportUninitializedInstanceVariable=false, reportUnknownMemberType=false, reportAttributeAccessIssue=false, reportUnknownVariableType=false, reportArgumentType=false, reportUnusedParameter=false, reportUnknownParameterType=false
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from typing_extensions import override

if TYPE_CHECKING:
    from narwhals.typing import IntoDataFrame
    from pfund.datas.timeframe import Timeframe
    from pfund.engines.backtest_engine import BacktestEngineContext
    from pfund.engines.settings.backtest_engine_settings import BacktestEngineSettings
    from pfund.entities.products.product_base import BaseProduct
    from pfund.typing import ComponentT
    from pfund.utils.dataset_splitter import (
        CrossValidatorDatasetPeriods,
        DatasetPeriods,
    )

from pfund.enums import BacktestMode


def setup_backtest_df(df: IntoDataFrame) -> IntoDataFrame:
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
    Calling setup_backtest_df starts a fresh session (clears the product
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
        raise ValueError(f"({', '.join(key_cols)}) keys must be unique for backtesting")
    # bars are processed positionally per (product, resolution) combo: an
    # unsorted df would silently produce garbage results
    if not nw_df.get_column("date").is_sorted():
        raise ValueError("the dataframe must be sorted by 'date' (ascending)")
    return df


class BacktestMixin:
    def __mixin_post_init__(self, *args: Any, **kwargs: Any):
        super().__mixin_post_init__(*args, **kwargs)
        self._cached_features_df: IntoDataFrame | None = None

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

    def backtest(self, df: IntoDataFrame) -> IntoDataFrame:
        if hasattr(super(), "backtest"):
            self._validate_backtest_signature(super().backtest)
            backtest_df: IntoDataFrame | None = cast(
                "IntoDataFrame | None", super().backtest(df)
            )
            if backtest_df is None:
                raise TypeError(
                    f"{self.name}.backtest() must return the backtested dataframe, got None. Did you forget to return df?"
                )
            return backtest_df
        else:
            if self.is_strategy():
                raise NotImplementedError(
                    f"{self.name} does not have a backtest(self, df) method, cannot run vectorized backtesting"
                )
            else:
                # model's backtest() is optional
                return df

    @property
    def context(self) -> BacktestEngineContext:
        assert self._context is not None, "context is not set"
        return cast("BacktestEngineContext", self._context)

    @property
    def settings(self) -> BacktestEngineSettings:
        return cast("BacktestEngineSettings", self.context.settings)

    @property
    def backtest_mode(self) -> BacktestMode:
        return self.context.backtest.backtest_mode

    @property
    def dataset_periods(self) -> DatasetPeriods | list[CrossValidatorDatasetPeriods]:
        return self.context.backtest.dataset_splitter.dataset_periods

    @property
    def features_df(self) -> IntoDataFrame | None:
        if self._cached_features_df is not None:
            return self._cached_features_df
        df = super().features_df
        # when components' features are not yet computed, it is not useful to cache the features_df since features_df = data_df + (empty signals_df)
        components_features_not_ready = (
            self.backtest_mode == BacktestMode.EXACT and not self.settings.reuse_signals
        )
        if (
            self.settings.cache_features_df
            and not components_features_not_ready
            and df is not None
        ):
            self._cached_features_df = df
        return df

    df = features_df

    # TODO
    @property
    def train_set(self):
        # FIXME: should use pfeed's config?
        storage_config = BacktestEngine._storage_config
        return self.store.load_data_from_storage(
            storage=storage_config.storage,
            storage_options=storage_config.storage_options,
        )

    # TODO
    @property
    def dev_set(self):
        return self.store.load_data(...)

    val_set = dev_set

    # TODO
    @property
    def test_set(self):
        return self.store.load_data(...)

    def _add_component(
        self,
        component: ComponentT,
        resolution: str = "",
        name: str = "",
        # NOTE: non-backtesting kwargs are ignored, e.g. ray_actor_options, ray_kwargs, etc.
        **kwargs: Any,
    ) -> ComponentT | None:
        from pfund.components.models.model_backtest import BacktestModel

        Component = type(component)
        component = BacktestModel(
            Component,
            component.model,
            *component.__pfund_args__,
            **component.__pfund_kwargs__,
        )
        return super()._add_component(
            component=component,
            resolution=resolution,
            name=name or Component.__name__,
        )

    def _is_dummy_strategy(self) -> bool:
        from pfund.components.strategies._dummy_strategy import _DummyStrategy

        return isinstance(self, _DummyStrategy)

    def _is_signals_precomputed(self) -> bool:
        return (
            not self.is_top_component() and self.backtest_mode == BacktestMode.FAST
        ) or (self.backtest_mode == BacktestMode.EXACT and self.settings.reuse_signals)

    def _materialize(self):
        for data_store in self.data_stores.values():
            data_store.materialize()
        if self._is_signals_precomputed():
            self.store.materialize()

    def _gather(self):
        if self._is_dummy_strategy():
            for component in self.components:
                component._gather()
            self._is_gathered = True
        else:
            return super()._gather()

    @override
    def get_supported_resolutions(
        self, product: BaseProduct
    ) -> dict[Timeframe, list[int]]:  # pyright: ignore[reportGeneralTypeIssues]
        """Gets supported resolutions for the product based on the trading venue.
        Overrides it in backtesting, supports only the primary resolution.
        """
        return {self.resolution.timeframe: [self.resolution.period]}
