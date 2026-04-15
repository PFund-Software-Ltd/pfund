# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportAttributeAccessIssue=false, reportArgumentType=false, reportUnknownArgumentType=false
from __future__ import annotations
from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    from narwhals._native import NativeDataFrame
    from pfeed.sources.pfund.component_feed import ComponentFeed
    from pfund.datas.databoy import DataBoy
    from pfund.typing import ComponentName

import logging

import narwhals as nw

from pfeed.enums import DataCategory, DataLayer, IOFormat, DataTool


class TradingStore:
    '''
    A TradingStore is a store that contains all data used by a component (e.g. strategy) in trading, from market data, computed features, to model predictions etc.
    '''
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
            
    # FIXME: add columns
    def _initialize_df(self) -> nw.DataFrame[Any]:
        context = self._databoy._component.context
        pfeed_config = context.pfeed_config
        if pfeed_config.data_tool == DataTool.polars:
            import polars as pl
            return nw.from_native(pl.DataFrame())
        elif pfeed_config.data_tool == DataTool.pandas:
            import pandas as pd
            return nw.from_native(pd.DataFrame())
        else:
            raise ValueError(f"Unsupported data tool: {pfeed_config.data_tool}")
    
    @property
    def df(self) -> nw.DataFrame[Any]:
        assert self._df is not None, "df is not set"
        return self._df
    
    # TODO: if df is not set, create an empty df
    def append_to_df(self):
        # self._initialize_df()
        pass
    
    def materialize(self):
        component = self._databoy._component
        # NOTE: component's signals_df (component.get_df()) should be ready before materialization, 
        # i.e. The component tree is BOTTOM-UP
        signals_dfs: dict[ComponentName, NativeDataFrame] = {
            _component.name: _component.get_df()
            for _component in component.get_components()
        }
        if not signals_dfs:
            return
        data_dfs: dict[DataCategory, NativeDataFrame] = {}
        for category in self._databoy.data_stores.keys():
            data_dfs[category] = self._databoy.get_df(kind='data', category=category)
        data_df = component.merge_data_dfs(data_dfs)
        features_df = component.featurize(data_df, signals_dfs)
        signals_df = component.signalize(features_df)
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
