# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportAttributeAccessIssue=false, reportArgumentType=false, reportUnknownArgumentType=false
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from narwhals.typing import Frame
    from pfund.engines.engine_context import EngineContext
    from pfeed.sources.pfund.component_feed import ComponentFeed

import logging

from pfeed.enums import DataLayer, IOFormat


class TradingStore:
    '''
    A TradingStore is a store that contains all data used by a component (e.g. strategy) in trading, from market data, computed features, to model predictions etc.
    '''
    def __init__(self, context: EngineContext, warmup_period: int | None=None):
        '''
        Args:
            warmup_period (int | None): Minimum number of data rows required before the component can produce signals.
                When `preload_min_data` is enabled in engine settings, these rows are pre-loaded during materialization
                for event-driven backtesting so the component starts warm.
                Defaults to 1 if None.
        '''
        import pfeed as pe
        self._logger: logging.Logger = logging.getLogger('pfund')
        self._context: EngineContext = context
        self._warmup_period: int = warmup_period if warmup_period else 1
        self._df: Frame | None = None  # signal df, wide form
        self._df_updates = []
        self._feed: ComponentFeed = pe.PFund().component_feed
        self._setup_component_feed()
        
    def _setup_component_feed(self):
        pfund_config = self._context.pfund_config
        for storage, storage_options in pfund_config.storage_options.items():
            self._feed.configure_storage(storage=storage, storage_options=storage_options)
        for io_format, io_options in pfund_config.io_options.items():
            self._feed.configure_io(io_format=io_format, io_options=io_options)
    
    @property
    def df(self) -> Frame:
        assert self._df is not None, "df is not set"
        return self._df
    
    @df.setter
    def df(self, df: Frame):
        self._df = df
    
    def materialize(self):
        for data_store in self._data_stores.values():
            data_store.materialize()

    def _persist_to_lakehouse(self):
        '''Load pfund's component (strategy/model/feature/indicator) data, e.g. {strategy_name}.parquet, {model_name}.parquet, etc.
        from the online store (TradingStore) to the offline store (pfeed's data lakehouse).
        '''
        pfund_config = self._context.pfund_config
        pfeed_config = self._context.pfeed_config
        
        data_layer = DataLayer.CURATED
        io_format = IOFormat.PARQUET

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
    