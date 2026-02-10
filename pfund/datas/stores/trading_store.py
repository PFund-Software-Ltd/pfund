from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund.engines.engine_context import EngineContext
    from pfeed.typing import GenericFrame, GenericSeries
    from pfeed.storages.base_storage import BaseStorage
    from pfeed.sources.pfund.component_feed import PFundComponentFeed
    from pfeed.sources.pfund.component_data_model import PFundComponentDataModel
    from pfund.datas.stores.base_data_store import BaseDataStore
    from pfund.datas.data_time_based import TimeBasedData

import logging

import pfeed as pe
from pfeed.enums import DataStorage, DataCategory, DataLayer, IOFormat
from pfund.datas.storage_config import StorageConfig
from pfund.datas.stores.market_data_store import MarketDataStore


class TradingStore:
    '''
    A TradingStore is a store that contains all data used by a component (e.g. strategy) in trading, from market data, computed features, to model predictions etc.
    '''
    def __init__(self, context: EngineContext):
        self._logger: logging.Logger = logging.getLogger('pfund')
        self._context: EngineContext = context
        self._feed: PFundComponentFeed = pe.PFund(env=context.env).component_feed
        # TEMP
        # self._storage: BaseStorage = self._create_storage()
        self._data_stores: dict[DataCategory, BaseDataStore] = {}
        self._df: GenericFrame | None = None
        self._df_updates = []
    
    # TODO
    def _prepare_df(self) -> GenericFrame:
        pass
    
    @property
    def market_data_store(self) -> MarketDataStore:
        return self.get_data_store(DataCategory.MARKET_DATA)
    market = market_data_store

    def _create_data_store(self, category: DataCategory) -> BaseDataStore:
        if category == DataCategory.MARKET_DATA:
            return MarketDataStore(self._context)
        else:
            raise ValueError(f'{category} is not supported')
        
    def get_data_store(self, category: DataCategory) -> BaseDataStore:
        if category in self._data_stores:
            return self._data_stores[category]
        else:
            return self._create_data_store(category)

    def _create_storage(self) -> BaseStorage:
        '''Create storage for component data to store signal dfs.
        e.g. {strategy_name}.parquet, {model_name}.parquet, etc.
        '''
        config = self._context.config
        settings = self._context.settings
        storage = DataStorage[config.database]
        storage_config: StorageConfig = StorageConfig(
            data_path=config.data_path,
            data_layer=DataLayer.CURATED,
            data_domain='trading_store',
            storage=DataStorage.LOCAL,
            io_format=IOFormat.PARQUET,
        )
        storage_options: dict = settings.storage_options.get(storage, {})
        io_options: dict = settings.io_options.get(storage_config.io_format, {})
        Storage = storage.storage_class
        # TODO
        data_model: PFundComponentDataModel = self._feed.create_data_model(...)
        return (
            Storage(
                data_path=storage_config.data_path,
                data_layer=storage_config.data_layer,
                data_domain=storage_config.data_domain,
                storage_options=storage_options,
            )
            .with_data_model(data_model)
            .with_io(
                io_options=io_options,
                io_format=storage_config.io_format,
            )
        )

    def get_market_data_df(self, data: TimeBasedData | None=None, unstack: bool=False) -> GenericFrame | None:
        if data is None:
            return self.market.data
        else:
            # TODO: filter data based on data.product and data.resolution
            return self.market.data
    
    def get_complete_df(self) -> GenericFrame | None:
        pass
    
    def get_strategy_df(
        self, 
        name: str='', 
        include_data: bool=False,
        as_series: bool=False,
    ) -> GenericFrame | GenericSeries | None:
        '''
        Get the dataframe of the strategy's outputs.
        Args:
            name: the name of the strategy
            include_data: whether to include the data dataframe in the output dataframe
                if not, only returns the strategy's outputs as a dataframe
            as_series: whether to return the dataframe as a series
        '''
        pass
    
    def get_model_df(
        self, 
        name: str='', 
        include_data: bool=False,
        as_series: bool=False,
    ) -> GenericFrame | GenericSeries | None:
        pass
    
    def get_indicator_df(
        self, 
        name: str='', 
        include_data: bool=False,
        as_series: bool=False,
    ) -> GenericFrame | GenericSeries | None:
        pass
     
    def get_feature_df(
        self, 
        name: str='', 
        include_data: bool=False,
        as_series: bool=False,
    ) -> GenericFrame | GenericSeries | None:
        pass
    
    def _get_df(self) -> GenericFrame | None:
        pass

    def materialize(self):
        for data_store in self._data_stores.values():
            data_store.materialize()
        
    # TODO: I/O should be async
    def _write_to_storage(self, data: GenericFrame):
        '''Load pfund's component (strategy/model/feature/indicator) data 
        from the online store (TradingStore) to the offline store (pfeed's data lakehouse).
        '''
        import pfeed as pe        

        
        data_model: PFundComponentDataModel = self._feed.create_data_model(...)
        
        data_layer = DataLayer.CURATED
        data_domain = 'trading_data'
        metadata = {}  # TODO
        storage: BaseStorage = pe.create_storage(
            storage=self._storage.value,
            data_model=data_model,
            data_layer=data_layer.value,
            data_domain=data_domain,
            storage_options=self._storage_options,
        )
        storage.write_data(data, metadata=metadata)
        self._logger.info(f'wrote {data_model} data to {storage.name} in {data_layer=} {data_domain=}')

    # TODO: when pfeed's data recording is ready
    def _rehydrate_from_pfeed(self):
        '''
        Load data from pfeed's data lakehouse if theres missing data after backfilling.
        '''
        pass
    