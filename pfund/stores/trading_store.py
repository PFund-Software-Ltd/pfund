from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import datetime
    from logging import Logger
    from pfeed.enums import DataTool, DataStorage
    from pfeed._typing import GenericFrame, GenericSeries
    from pfeed.storages.base_storage import BaseStorage
    from pfeed.sources.pfund.engine_feed import PFundEngineFeed
    from pfeed.sources.pfund.data_model import PFundDataModel
    from pfund.datas.data_time_based import TimeBasedData
    from pfund._typing import DataParamsDict

from pfeed.enums import DataCategory, DataLayer
from pfund.enums import Environment
from pfund.stores.market_data_store import MarketDataStore


class TradingStore:
    '''
    A TradingStore is a store that contains all data used in trading, from market data, computed features, to model predictions etc.
    '''
    def __init__(self, env: Environment, data_params: DataParamsDict):
        from pfeed.sources.pfund import PFund

        self._logger = None
        self._env = env
        self._data_params: DataParamsDict = data_params
        
        
        # FIXME
        # self._feed: PFundEngineFeed = PFund(
        #     env=env.value,
        #     data_tool=data_tool.value,
        #     use_ray=False,  # FIXME
        #     use_deltalake=True,
        # )
        self._feed = None
        
        self._data_stores = {
            DataCategory.MARKET_DATA: MarketDataStore(
                data_tool=data_params['data_tool'],
                storage=data_params['storage'],
                storage_options=data_params['storage_options'],
                feed=self._feed,
            ),
        }
        self._df: GenericFrame | None = None
        self._df_updates = []
        
    @property
    def market_data_store(self) -> MarketDataStore:
        return self._data_stores[DataCategory.MARKET_DATA]
    market = market_data_store

    def _set_logger(self, logger: Logger):
        self._logger = logger
        for store in self._data_stores.values():
            store._set_logger(logger)
    
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

    def _materialize(self):
        for data_store in self._data_stores.values():
            data_store._materialize()
        
    def _write_to_storage(self, data: GenericFrame):
        '''
        Load pfund's component (strategy/model/feature/indicator) data from the online store (TradingStore) to the offline store (pfeed's data lakehouse).
        '''
        from pfeed import create_storage
        
        data_model: PFundDataModel = self._feed.create_data_model(...)
        data_layer = DataLayer.CURATED
        data_domain = 'trading_data'
        metadata = {}  # TODO
        storage: BaseStorage = create_storage(
            storage=self._storage.value,
            data_model=data_model,
            data_layer=data_layer.value,
            data_domain=data_domain,
            storage_options=self._storage_options,
        )
        storage.write_data(data, metadata=metadata)
        self._logger.info(f'wrote {data_model} data to {storage.name} in {data_layer=} {data_domain=}')

    # TODO: when pfeed's data recording is ready
    def _rehydrate_from_lakehouse(self):
        '''
        Load data from pfeed's data lakehouse if theres missing data after backfilling.
        '''
        pass
    