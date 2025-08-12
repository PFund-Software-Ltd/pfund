from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfeed.enums import DataTool, DataStorage
    from pfeed._typing import GenericData
    from pfeed.sources.pfund.engine_feed import PFundEngineFeed
    from pfeed.sources.pfund.data_model import PFundDataModel
    from pfeed.storages.base_storage import BaseStorage
    
from abc import ABC, abstractmethod
from logging import Logger

from pfeed import create_storage
from pfeed.enums import DataLayer


class BaseDataStore(ABC):
    def __init__(
        self, 
        data_tool: DataTool,
        storage: DataStorage,
        storage_options: dict,
        feed: PFundEngineFeed,
    ):
        self._data_tool = data_tool
        self._storage = storage
        self._storage_options = storage_options
        self._logger: Logger | None = None
        self._feed: PFundEngineFeed = feed
        self._data: GenericData | None = None
        self._data_updates = []
        
    @abstractmethod
    def materialize(self, *args, **kwargs):
        pass

    @abstractmethod
    def _get_historical_data(self, *args, **kwargs) -> GenericData:
        pass

    @property
    def data(self) -> GenericData | None:
        if self._data is None:
            # TODO: read from storage if data is not in memory
            pass
        else:
            return self._data
    
    def _set_logger(self, logger: Logger):
        self._logger = logger
        
    def _set_data(self, data: GenericData):
        self._data = data
    
    # TODO: I/O should be async
    def _write_to_storage(self, data: GenericData):
        '''
        Load data (e.g. market data) from the online store (TradingStore) to the offline store (pfeed's data lakehouse).
        '''
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
