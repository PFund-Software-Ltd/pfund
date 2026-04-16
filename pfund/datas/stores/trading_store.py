# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportAttributeAccessIssue=false, reportArgumentType=false, reportUnknownArgumentType=false
from __future__ import annotations
from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    from narwhals._native import NativeDataFrame
    from pfeed.sources.pfund.component_feed import ComponentFeed
    from pfund.datas.databoy import DataBoy

import logging

import narwhals as nw

from pfeed.enums import DataLayer, IOFormat


class TradingStore:
    def __init__(self, databoy: DataBoy):
        import pfeed as pe
        self._logger: logging.Logger = logging.getLogger('pfund')
        self._databoy: DataBoy = databoy
        self._df: nw.DataFrame[Any] | None = None  # component's signals_df
        self._feed: ComponentFeed = pe.PFund().component_feed
        self._setup_component_feed()
        
    def _setup_component_feed(self):
        context = self._databoy._component.context
        pfund_config = context.pfund_config
        for storage, storage_options in pfund_config.storage_options.items():
            self._feed.configure_storage(storage=storage, storage_options=storage_options)
        for io_format, io_options in pfund_config.io_options.items():
            self._feed.configure_io(io_format=io_format, io_options=io_options)
            
    def get_df(self, window_size: int | None = None, to_native: bool = False) -> nw.DataFrame[Any] | NativeDataFrame | None:
        if self._df is None:
            return None
        df = self._df if window_size is None else self._df.tail(window_size)
        return df.to_native() if to_native else df
    
    def update_df(self, signals_df: nw.DataFrame[Any]):
        if self._df is None:
            self._df = signals_df
        else:
            self._df = nw.concat([self._df, signals_df], how='vertical')
        # TODO: trim df if it's too large
        # max_rows = self._databoy._component.config['max_rows']

    def pivot_df(self, df: nw.DataFrame[Any]) -> nw.DataFrame[Any]:
        '''Pivots signals dataframe from long form to wide form.
        Args:
            df: signals_df in long form
        '''
        component = self._databoy._component
        pivot_cols = [col for col in component._pivot_cols if col in df.columns]
        if not pivot_cols:
            raise ValueError(
                f"Cannot pivot component '{component.name}' signals_df to wide form: " +
                f"none of {component.name}'s pivot_cols={component._pivot_cols} appear in signals_df columns={df.columns}. " +
                "Please call set_pivot_cols() for your component to set the pivot columns properly."
            )
        index_cols = [col for col in component._index_cols if col in df.columns]
        return (
            df
            .pivot(
                on=pivot_cols,
                index=index_cols,
            )
            .sort(index_cols)
        )
    
    def materialize(self):
        component = self._databoy._component
        # NOTE: lookback_period=None means run the pipeline on the whole dataset
        signals_df = component.run_pipeline(lookback_period=None)
        self._df = nw.from_native(signals_df)
    
    def persist_to_lakehouse(self):
        '''Load pfund's component (strategy/model/feature/indicator) data, e.g. {strategy_name}.parquet, {model_name}.parquet, etc.
        from the online store (TradingStore) to the offline store (pfeed's data lakehouse).
        '''
        context = self._databoy._component.context
        pfund_config = context.pfund_config
        pfeed_config = context.pfeed_config
        
        data_layer = DataLayer.CURATED
        io_format = IOFormat.PARQUET

        # TODO: how to write updates? need to use deltalake
        self._feed.load(  # pyright: ignore[reportCallIssue]
            data=self._df,
            storage=pfund_config.storage,
            data_path=pfeed_config.data_path,
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
        self._feed.retrieve(...)
    
    # TODO:
    def swap_live_for_eod(self):
        '''Discard the interim live-stream buffer and load the official end-of-day dataset (if any).'''
        pass
