from __future__ import annotations
from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    from pfeed.enums import DataSource, DataCategory
    from pfund.datas.data_config import DataConfig
    from pfeed.storages.storage_config import StorageConfig

from abc import ABC, abstractmethod


class BaseData(ABC):
    category: DataCategory
    
    def __init__(
        self, 
        data_source: DataSource, 
        data_origin: str, 
        data_config: DataConfig,
        storage_config: StorageConfig,
    ):
        self.source: DataSource = data_source
        self.origin: str = data_origin
        self.config: DataConfig = data_config
        self.storage_config: StorageConfig = storage_config
        self.extra_data: dict[str, Any] = {}
        
    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        pass
    
    def update_extra_data(self, extra_data: dict[str, Any]):
        self.extra_data = extra_data
    
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, BaseData):
            return NotImplemented
        return (
            self.source == other.source
            and self.origin == other.origin
        )
    
    def __hash__(self):
        return hash((self.source, self.origin))