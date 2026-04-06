from __future__ import annotations
from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    from narwhals.typing import Frame
    from pfund.engines.engine_context import EngineContext

import logging
from abc import ABC, abstractmethod

from pfeed.feeds.base_feed import BaseFeed
from pfeed.enums import DataTool, DataStorage, DataLayer
from pfeed.storages.storage_config import StorageConfig
from pfund.datas.data_base import BaseData


DataT = TypeVar('DataT', bound=BaseData)
FeedT = TypeVar('FeedT', bound=BaseFeed)


class BaseDataStore(ABC, Generic[DataT, FeedT]):
    SUPPORTED_DATA_TOOLS = [DataTool.pandas, DataTool.polars]
    
    def __init__(self, context: EngineContext):
        self._logger: logging.Logger = logging.getLogger("pfund")
        self._data_tool = context.pfeed_config.data_tool
        if self._data_tool not in self.SUPPORTED_DATA_TOOLS:
            raise ValueError(f"Unsupported data tool: {self._data_tool}")
        self._context: EngineContext = context
        self._df: Frame | None = None  # data df, long form
        self._df_updates = []
        self._feeds: dict[DataT, FeedT] = {}
        self._storage_configs: dict[DataT, StorageConfig] = {}

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

    @property
    def df(self) -> Frame:
        assert self._df is not None, "df is not set"
        return self._df
    
    @df.setter
    def df(self, df: Frame):
        self._df = df
        
    def get_datas(self) -> list[DataT]:
        return list(self._feeds.keys())

    def _create_feed(self, data: DataT) -> FeedT:
        from pfeed.feeds import create_feed
        return create_feed(  # pyright: ignore[reportReturnType]
            data_source=data.source,
            data_category=data.category,
            pipeline_mode=True,
            num_workers=self._context.settings.num_workers,
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
