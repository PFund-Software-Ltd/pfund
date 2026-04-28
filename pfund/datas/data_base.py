from __future__ import annotations
from typing import Any, ClassVar
    
from pfeed.enums import DataLayer, DataSource, DataCategory
from pfeed.storages.storage_config import StorageConfig
from pfund.datas.data_config import DataConfig


class BaseData:
    category: ClassVar[DataCategory]
    
    def __init__(
        self, 
        data_config: DataConfig | None=None, 
        storage_config: StorageConfig | None=None,
    ):
        self.config: DataConfig = data_config or DataConfig()
        self.storage_config: StorageConfig = storage_config or StorageConfig()
        if self.storage_config.data_layer == DataLayer.RAW:
            raise ValueError('data from RAW data layer is not supported, pfund can only deal with cleaned data')
        self.extra_data: dict[str, Any] = {}
    
    @property
    def source(self) -> DataSource:
        assert self.config.data_source is not None
        return DataSource[self.config.data_source.upper()]
    
    @property
    def origin(self) -> str:
        return self.config.data_origin
        
    def to_dict(self) -> dict[str, Any]:
        return {
            'data_config': self.config.model_dump(),
            'storage_config': self.storage_config.model_dump(),
        }
    
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