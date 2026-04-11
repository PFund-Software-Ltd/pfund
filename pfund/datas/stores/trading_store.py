# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportAttributeAccessIssue=false, reportArgumentType=false, reportUnknownArgumentType=false
from __future__ import annotations
from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    from pfeed.sources.pfund.component_feed import ComponentFeed
    from pfund.engines.engine_context import EngineContext

import logging

import narwhals as nw

from pfeed.enums import DataCategory, DataLayer, IOFormat


class TradingStore:
    '''
    A TradingStore is a store that contains all data used by a component (e.g. strategy) in trading, from market data, computed features, to model predictions etc.
    '''
    def __init__(self, context: EngineContext):
        import pfeed as pe
        self._logger: logging.Logger = logging.getLogger('pfund')
        self._context: EngineContext = context
        self._df: nw.DataFrame[Any] | None = None  # component's signals_df
        self._feed: ComponentFeed = pe.PFund().component_feed
        self._setup_component_feed()
        
    def _setup_component_feed(self):
        pfund_config = self._context.pfund_config
        for storage, storage_options in pfund_config.storage_options.items():
            self._feed.configure_storage(storage=storage, storage_options=storage_options)
        for io_format, io_options in pfund_config.io_options.items():
            self._feed.configure_io(io_format=io_format, io_options=io_options)
    
    @property
    def df(self) -> nw.DataFrame[Any]:
        assert self._df is not None, "df is not set"
        return self._df
    
    def _set_signal_df(self, signal_df: pd.DataFrame | pl.LazyFrame):
        assert signal_df.shape[0] == self.df.shape[0], f"{signal_df.shape[0]=} != {self.df.shape[0]=}"
        nan_columns = self.data_tool.get_nan_columns(signal_df)
        assert not nan_columns, f"{self.name} signal_df has all NaN values in columns: {nan_columns}"
        self._signal_list = signal_df.drop(columns=self.INDEX).to_numpy().tolist()
        self._signal_df = signal_df

    def materialize(self, data_dfs: dict[DataCategory, nw.DataFrame[Any]], signals_df: nw.DataFrame[Any]):
        dfs = list(data_dfs.values())
        data_df = dfs[0]
        for df in dfs[1:]:
            data_df = data_df.join(df, on=['date'], how='full')
        self._df = data_df.join(signals_df, on=['date'], how='full')
    
    def _persist_to_lakehouse(self):
        '''Load pfund's component (strategy/model/feature/indicator) data, e.g. {strategy_name}.parquet, {model_name}.parquet, etc.
        from the online store (TradingStore) to the offline store (pfeed's data lakehouse).
        '''
        pfund_config = self._context.pfund_config
        pfeed_config = self._context.pfeed_config
        
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
    def _rehydrate_from_lakehouse(self):
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
