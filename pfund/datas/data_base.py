from __future__ import annotations
from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    from pfeed.enums import DataSource, DataCategory

from abc import ABC, abstractmethod


class BaseData(ABC):
    def __init__(self, data_source: DataSource, data_origin: str=''):
        self.source: DataSource = data_source
        self.origin: str = data_origin
        self._extra_data: dict[str, Any] = {}
        self._custom_data: dict[Any, Any] = {}
        
    @abstractmethod
    def to_dict(self) -> dict:
        pass
    
    @property
    @abstractmethod
    def category(self) -> DataCategory:
        pass
    
    @property
    def extra_data(self):
        return self._extra_data
    
    @property
    def custom_data(self):
        return self._custom_data
    
    def update_extra_data(self, extra_data: dict):
        self._extra_data = extra_data
    
    def update_custom_data(self, custom_data: dict):
        self._custom_data = custom_data

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, BaseData):
            return NotImplemented
        return (
            self.source == other.source
            and self.origin == other.origin
        )
    
    def __hash__(self):
        return hash((self.source, self.origin))