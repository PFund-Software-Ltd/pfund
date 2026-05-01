# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportAttributeAccessIssue=false, reportArgumentType=false, reportUnknownArgumentType=false, reportOptionalMemberAccess=false, reportConstantRedefinition=false
from __future__ import annotations
from typing import TYPE_CHECKING, Any, ClassVar
if TYPE_CHECKING:
    from narwhals._native import NativeDataFrame
    from pfeed.sources.pfund.component_feed import ComponentFeed
    from pfund.typing import ComponentName, Component, ColumnName

import narwhals as nw

from pfeed.storages.storage_config import StorageConfig
from pfeed.enums import DataLayer, IOFormat
from pfund.enums import RunStage


class TradingStore:
    INDEX_COL: ClassVar[str] = 'date'
    PIVOT_COLS: ClassVar[list[str]] = ['product', 'resolution']
    
    def __init__(self, component: Component):
        import pfeed as pe
        self._component: Component = component
        self._df: nw.DataFrame[Any] | None = None  # component's signals_df
        self._feed = pe.PFund().component_feed(self._component)
        context = component.context
        self._storage_config = StorageConfig(
            data_path=context.pfund_config.data_path,
            # REVIEW: data_layer, data_domain are NOT in use.
            data_layer={
                RunStage.EXPERIMENT: DataLayer.RAW,
                RunStage.REFINEMENT: DataLayer.CLEANED,
                RunStage.DEPLOYMENT: DataLayer.CURATED,
            }[context.run_stage],
            data_domain='PFUND_DATA',
        )
    
    @property
    def KEY_COLS(self) -> list[str]:
        return [self.INDEX_COL] + self.PIVOT_COLS
    
    @property
    def logger(self):
        return self._component.logger
    
    @property
    def component_feed(self) -> ComponentFeed:
        return self._feed
        
    def set_pivot_cols(self, pivot_cols: list[str]):
        self.PIVOT_COLS = pivot_cols
    
    def set_storage_config(self, storage_config: StorageConfig):
        self._storage_config = storage_config
    
    def get_df(
        self,
        window_size: int | None = None,
        to_native: bool = False,
    ) -> nw.DataFrame[Any] | NativeDataFrame | None:
        '''
        Args:
            window_size: Number of most recent rows to return.
            to_native: If True, return the underlying backend frame (polars/pandas) instead
                of a Narwhals DataFrame. Defaults to True.
        '''
        df = self._df
        if df is None:
            return None
        if window_size is not None:
            df = df.tail(window_size)
        return df.to_native() if to_native else df
    
    def update_df(self, features_df: nw.DataFrame[Any], signals: dict[ColumnName, Any]):
        '''
        Args:
            features_df: features used to compute signals in dataframe form
            signals: computed signals
        '''
        df_backend = nw.get_native_namespace(features_df)
        signals_df = nw.concat([
            features_df.select(self.key_cols),
            nw.DataFrame.from_dict(data=signals, backend=df_backend),
        ], how='horizontal')
        if self._df is None:
            self._df = signals_df
        else:
            self._df = nw.concat([self._df, signals_df], how='vertical')
        max_rows = self._component.config['max_rows']
        if max_rows is not None and len(self._df) > max_rows:
            self._df = self._df.tail(max_rows)

    def pivot_df(self, df: nw.DataFrame[Any]) -> nw.DataFrame[Any]:
        '''Pivots signals dataframe from long form to wide form.
        Args:
            df: signals_df in long form
        '''
        return (
            df
            .pivot(
                on=self.PIVOT_COLS,
                index=self.INDEX_COL,
            )
            .sort(self.INDEX_COL)
        )
    
    # TODO: load {component_name}.parquet
    def materialize(self):
        # NOTE: lookback_period=None means run the pipeline on the whole dataset
        # self._component.run_pipeline(lookback_period=None)
        # self._component.load() ???
        pass
    
    # TODO
    def persist_to_lakehouse(self):
        '''Load pfund's component (strategy/model/feature/indicator) data, e.g. {strategy_name}.parquet, {model_name}.parquet, etc.
        from the online store (TradingStore) to the offline store (pfeed's data lakehouse).
        '''
        context = self._component.context
        pfund_config = context.pfund_config
        
        data_layer = DataLayer.CURATED
        io_format = IOFormat.PARQUET

        # TODO: how to write updates? need to use deltalake
        self.component_feed.load(
            data=self._df,
            storage=self._storage_config.storage,
            data_path=pfund_config.data_path,
            data_layer=data_layer,
            io_format=io_format,
        )

    # TODO:
    def rehydrate_from_lakehouse(self):
        '''
        Load data from pfeed's data lakehouse when:
        - theres missing data
        - EOD data is available, replace SourceType.STREAM (live data) with SourceType.BATCH (official EOD data)
        '''
        self.component_feed.retrieve(...)
    
    # TODO:
    def swap_live_for_eod(self):
        '''Discard the interim live-stream buffer and load the official end-of-day dataset (if any).'''
        pass
