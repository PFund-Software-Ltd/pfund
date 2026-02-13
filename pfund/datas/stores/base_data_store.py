from __future__ import annotations
from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    from pfeed.typing import GenericData
    from pfund.engines.engine_context import EngineContext

import logging
from abc import ABC, abstractmethod

from pfeed.feeds.base_feed import BaseFeed
from pfeed.enums import DataTool
from pfeed import get_config as get_pfeed_config
from pfeed.storages.storage_config import StorageConfig
from pfund.datas.data_base import BaseData


DataT = TypeVar('DataT', bound=BaseData)
FeedT = TypeVar('FeedT', bound=BaseFeed)


pfeed_config = get_pfeed_config()


class BaseDataStore(ABC, Generic[DataT, FeedT]):
    SUPPORTED_DATA_TOOLS = [DataTool.pandas, DataTool.polars]
    
    def __init__(self, context: EngineContext):
        self._logger: logging.Logger = logging.getLogger("pfund")
        self._data_tool = pfeed_config.data_tool
        if self._data_tool not in self.SUPPORTED_DATA_TOOLS:
            raise ValueError(f"Unsupported data tool: {self._data_tool}")
        self._context: EngineContext = context
        self._feeds: dict[DataT, FeedT] = {}
        self._storage_configs: dict[BaseData, StorageConfig] = {}

    @abstractmethod
    def materialize(self, *args: Any, **kwargs: Any):
        pass

    def _create_feed(self, data: DataT) -> FeedT:
        from pfeed.feeds import create_feed
        return create_feed(  # pyright: ignore[reportReturnType]
            data_source=data.source,
            data_category=data.category,
            pipeline_mode=True,
            num_batch_workers=self._context.settings.num_batch_workers,
            num_stream_workers=None,
        )

    def add_data(self, data: DataT, storage_config: StorageConfig | None=None):
        storage_config = storage_config or StorageConfig()
        self._storage_configs[data] = storage_config
        self._feeds[data] = self._create_feed(data)
