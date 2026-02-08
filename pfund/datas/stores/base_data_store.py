from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pfeed.typing import GenericData
    from pfeed.feeds import BaseFeed
    from pfund.engines.engine_context import EngineContext
    from pfund.datas.storage_config import StorageConfig
    from pfund.datas.data_base import BaseData
    
import logging
from abc import ABC, abstractmethod


class BaseDataStore(ABC):
    def __init__(self, context: EngineContext):
        self._logger = logging.getLogger('pfund')
        self._context: EngineContext = context
        self._feeds: dict[BaseData, BaseFeed] = {}
        self._storage_configs: dict[BaseData, StorageConfig] = {}

    @abstractmethod
    def materialize(self, *args, **kwargs):
        pass

    @abstractmethod
    def _get_historical_data(self, *args, **kwargs) -> GenericData:
        pass

    def _create_feed(self, data: BaseData) -> BaseFeed:
        from pfeed.feeds import create_feed
        return create_feed(
            data_source=data.source,
            data_category=data.category,
            pipeline_mode=True,
        )
    
    def add_data(self, data: BaseData, storage_config: StorageConfig | None):
        if storage_config is None:
            storage_config = StorageConfig()
        if not storage_config.data_domain:
            storage_config.data_domain = data.category
        self._storage_configs[data] = storage_config
        self._feeds[data] = self._create_feed(data)