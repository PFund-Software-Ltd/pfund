# pyright: reportUninitializedInstanceVariable=false, reportUnknownMemberType=false, reportAttributeAccessIssue=false, reportUnknownVariableType=false, reportArgumentType=false, reportUnusedParameter=false, reportUnknownParameterType=false
from __future__ import annotations
from typing_extensions import override
from typing import TYPE_CHECKING, cast, Callable, Any
if TYPE_CHECKING:
    from narwhals._native import NativeDataFrame
    from pfeed.typing import GenericFrame
    from pfund._backtest.typing import BacktestDataFrame
    from pfund.typing import ComponentT
    from pfund.entities.products.product_base import BaseProduct
    from pfund.datas.timeframe import Timeframe
    from pfund.engines.backtest_engine import BacktestEngineContext
    from pfund.engines.settings.backtest_engine_settings import BacktestEngineSettings
    from pfund.utils.dataset_splitter import DatasetPeriods, CrossValidatorDatasetPeriods

from pfund.enums import BacktestMode


class BacktestMixin:
    def __mixin_post_init__(self, *args: Any, **kwargs: Any):
        super().__mixin_post_init__(*args, **kwargs)
        self._cached_features_df: NativeDataFrame | None = None
    
    @staticmethod
    def _validate_backtest_signature(func: Callable[[BacktestDataFrame], BacktestDataFrame]):
        '''Validates the signature of the backtest() function.
        The backtest() function must accept exactly 1 argument (df) and return a BacktestDataFrame.
        '''
        import ast
        import inspect
        import textwrap

        sig = inspect.signature(func)
        params = [p for p in sig.parameters if p not in ('self', 'cls')]
        if len(params) != 1:
            raise TypeError(f"backtest() must accept exactly 1 argument (df), got {len(params)}: {params}")

        try:
            source = textwrap.dedent(inspect.getsource(func))
            tree = ast.parse(source)
            func_def = tree.body[0]
            has_return = any(isinstance(node, ast.Return) and node.value is not None for node in ast.walk(func_def))
            if not has_return:
                raise TypeError("backtest() must return a BacktestDataFrame. No return statement found. Did you forget to return df?")
        except OSError:
            pass  # source not available (e.g. built-in), skip check

    def backtest(self, df: BacktestDataFrame) -> BacktestDataFrame:
        if hasattr(super(), 'backtest'):
            self._validate_backtest_signature(getattr(super(), 'backtest'))
            backtest_df: BacktestDataFrame | None = cast("BacktestDataFrame | None", super().backtest(df))
            if backtest_df is None:
                raise TypeError(f"{self.name}.backtest() must return a BacktestDataFrame, got None. Did you forget to return df?")
            return backtest_df
        else:
            if self.is_strategy():
                raise NotImplementedError(f'{self.name} does not have a backtest(self, df) method, cannot run vectorized backtesting')
            else:
                # model's backtest() is optional
                return df
    
    @property
    def context(self) -> BacktestEngineContext:
        assert self._context is not None, 'context is not set'
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
    def features_df(self) -> NativeDataFrame | None:
        if self._cached_features_df is not None:
            return self._cached_features_df
        df = super().features_df
        # when components' features are not yet computed, it is not useful to cache the features_df since features_df = data_df + (empty signals_df)
        components_features_not_ready = (
            self.backtest_mode == BacktestMode.EVENT_DRIVEN 
            and not self.settings.reuse_signals
        )
        if self.settings.cache_features_df and not components_features_not_ready and df is not None:
            self._cached_features_df = df
        return df
    df = features_df
    
    # TODO
    @property
    def train_set(self) -> GenericFrame:
        # FIXME: should use pfeed's config?
        storage_config = BacktestEngine._storage_config
        return self.store.load_data_from_storage(
            storage=storage_config.storage,
            storage_options=storage_config.storage_options,
        )
    
    # TODO
    @property
    def dev_set(self) -> GenericFrame:
        return self.store.load_data(...)
    val_set = dev_set
    
    # TODO
    @property
    def test_set(self) -> GenericFrame:
        return self.store.load_data(...)
    
    def _add_component(
        self, 
        component: ComponentT,
        resolution: str='',
        name: str='', 
        # NOTE: non-backtesting kwargs are ignored, e.g. ray_actor_options, ray_kwargs, etc.
        **kwargs: Any,
    ) -> ComponentT | None:
        from pfund.components.models.model_backtest import BacktestModel
        Component = type(component)
        component = BacktestModel(Component, component.model, *component.__pfund_args__, **component.__pfund_kwargs__)
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
            (not self.is_top_component() and self.backtest_mode in [BacktestMode.VECTORIZED, BacktestMode.HYBRID])
            or (self.backtest_mode == BacktestMode.EVENT_DRIVEN and self.settings.reuse_signals)
        )
    
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
    def get_supported_resolutions(self, product: BaseProduct) -> dict[Timeframe, list[int]]:  # pyright: ignore[reportGeneralTypeIssues]
        '''Gets supported resolutions for the product based on the trading venue.
        Overrides it in backtesting, supports only the primary resolution.
        '''
        return {
            self.resolution.timeframe: [self.resolution.period]
        }