# pyright: reportUninitializedInstanceVariable=false, reportUnknownMemberType=false, reportAttributeAccessIssue=false, reportUnknownVariableType=false, reportArgumentType=false, reportUnusedParameter=false, reportUnknownParameterType=false
from __future__ import annotations
from typing import TYPE_CHECKING, cast, Callable, Any
if TYPE_CHECKING:
    from narwhals._native import NativeDataFrame
    from pfeed.typing import GenericFrame
    from pfund._backtest.typing import BacktestDataFrame
    from pfund.typing import ComponentT, ComponentName
    from pfund.entities.products.product_base import BaseProduct
    from pfund.components.mixin import ComponentMixin
    from pfund.engines.backtest_engine import BacktestEngineContext
    from pfund.engines.settings.backtest_engine_settings import BacktestEngineSettings
    from pfund.utils.dataset_splitter import DatasetPeriods, CrossValidatorDatasetPeriods

import narwhals as nw

from pfund_kit.style import cprint, RichColor, TextStyle
from pfund.datas.data_config import DataConfig
from pfund.enums import BacktestMode


class BacktestMixin:
    def __mixin_post_init__(self, *args: Any, **kwargs: Any):
        super().__mixin_post_init__(*args, **kwargs)
        self._features_df: nw.DataFrame[Any] | None = None
        
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
    
    def _is_features_df_required(self) -> bool:
        if self.backtest_mode in [BacktestMode.VECTORIZED, BacktestMode.HYBRID]:
            return True
        elif self.backtest_mode == BacktestMode.EVENT_DRIVEN:
            return self.settings.reuse_signals
        else:
            return False
    
    def featurize(self, data_df: NativeDataFrame, signals_dfs: dict[ComponentName, NativeDataFrame]) -> NativeDataFrame:
        features_df = cast("NativeDataFrame", super().featurize(data_df, signals_dfs))
        if self._is_features_df_required():
            self._features_df = nw.from_native(features_df)
        return features_df
    
    def _gather(self):
        if self._is_dummy_strategy():
            for component in self.components:
                component._gather()
            self._is_gathered = True
        else:
            return super()._gather()
    
    # FIXME: move into DataConfigResolver
    def _resolve_data_config(self, product: BaseProduct, data_config: DataConfig | None) -> DataConfig:
        data_config = cast("ComponentMixin", super())._resolve_data_config(product, data_config)
        # extra_resolutions are not always supported in backtesting, print out warnings
        if data_config.extra_resolutions:
            if self.backtest_mode != BacktestMode.EVENT_DRIVEN:
                cprint(
                    f'{product.name} extra_resolutions={data_config.extra_resolutions} will be ignored in {self.backtest_mode.value} backtesting',
                    style=TextStyle.BOLD + RichColor.RED
                )
            else:
                # REVIEW: support looping quote/tick data?
                if any(resolution.is_quote() or resolution.is_tick() for resolution in data_config.resolutions):
                    cprint(
                        f'WARNING: {product.name} tick/quote data will be ignored in backtesting',
                        style=TextStyle.BOLD + RichColor.RED
                    )
        return data_config
    
    # FIXME: move into DataConfigResolver
    def _auto_resample_data_config(self, product: BaseProduct, data_config: DataConfig) -> DataConfig:
        '''Automatically configures resampling for backtesting.

        Overrides ComponentMixin._auto_resample_data_config() to force resampling setup for
        any extra resolutions that are lower (less granular) than the primary resolution.

        Backtesting Constraint:
            In live trading, multiple data resolutions can be subscribed independently
            (e.g., both 1m and 1h data streams). However, in backtesting, only the primary
            resolution is used for event-driven simulation. Any lower resolutions must be
            generated by resampling/aggregating from the primary resolution.

        Resolution Comparison (pfund convention):
            - Higher resolution (more granular, e.g., '1m') is considered "greater"
            - Lower resolution (less granular, e.g., '1h') is considered "less"
            - Therefore: '1h' < '1m' evaluates to True

        Example:
            Given:  primary_resolution='1m', extra_resolutions=['1h']
            Result: resample={'1h': '1m'}
            Meaning: Aggregate sixty 1m bars into one 1h bar during backtesting

        Args:
            product: The product being backtested
            data_config: Configuration containing resolutions and resampling rules

        Returns:
            Updated DataConfig with automatic resampling configured

        Raises:
            Exception: If a lower resolution cannot be evenly resampled from the primary
                      resolution (e.g., '1h' cannot be cleanly resampled from '45m')
        '''
        # Only event-driven mode supports extra resolutions
        if self.backtest_mode != BacktestMode.EVENT_DRIVEN:
            return data_config  # Skip processing extra resolutions

        original_resample = data_config.resample.copy()
        primary_resolution = data_config.primary_resolution
        for resolution in data_config.resolutions:
            if resolution == primary_resolution:
                continue
            # REVIEW: support using tick data to resample bar data?
            if not resolution.is_bar():
                continue
            if resolution < primary_resolution:
                if resolution.to_seconds() % primary_resolution.to_seconds() == 0:
                    data_config.resample[resolution] = primary_resolution
                    self.logger.warning(
                        f'{product.name} is auto-resampled from {original_resample} to {data_config.resample} in backtesting'
                    )
                else:
                    raise Exception(f'{resolution=} is not supported in backtesting because it cannot be resampled by resolution={primary_resolution}')
            else:
                raise ValueError(f'extra resolution {resolution} higher than primary resolution {primary_resolution} is not allowed in backtesting')
        return data_config
