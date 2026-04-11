from __future__ import annotations
from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    from pfund.datas.databoy import DataBoy

import logging
from abc import ABC, abstractmethod

import narwhals as nw

from pfeed.feeds.base_feed import BaseFeed
from pfeed.enums import DataStorage, DataLayer
from pfeed.storages.storage_config import StorageConfig
from pfund.datas.data_base import BaseData


DataT = TypeVar('DataT', bound=BaseData)
FeedT = TypeVar('FeedT', bound=BaseFeed)


class BaseDataStore(ABC, Generic[DataT, FeedT]):
    def __init__(self, databoy: DataBoy):
        self._logger: logging.Logger = logging.getLogger("pfund")
        self._databoy: DataBoy = databoy
        self._df: nw.DataFrame[Any] | None = None

    @property
    def df(self) -> nw.DataFrame[Any]:
        if self._df is None:
            self.materialize()
        assert self._df is not None, "df is not set"
        return self._df

    @abstractmethod
    def materialize(self, *args: Any, **kwargs: Any) -> None:
        pass
    
    @abstractmethod
    def create_data(self, *args: Any, **kwargs: Any) -> DataT:
        pass

    @abstractmethod
    def add_data(self, *args: Any, **kwargs: Any) -> list[DataT]:
        pass
    
    @abstractmethod
    def get_data(self, *args: Any, **kwargs: Any) -> DataT | None:
        pass

    @abstractmethod
    def get_datas(self) -> list[DataT]:
        pass

    def _create_feed(self, data: DataT) -> FeedT:
        from pfeed.feeds import create_feed
        return create_feed(  # pyright: ignore[reportReturnType]
            data_source=data.source,
            data_category=data.category,
            pipeline_mode=True,
            num_workers=data.config.num_batch_workers,
        )
    
    def _create_cache_storage_config(self, storage_config: StorageConfig) -> StorageConfig:
        '''Create a cache storage config inheriting io settings from the original storage config.'''
        return StorageConfig(
            storage=DataStorage.CACHE,
            data_path=storage_config.data_path,
            data_layer=DataLayer.CURATED,
            io_format=storage_config.io_format,
            compression=storage_config.compression,
        )
