from __future__ import annotations

from typing import Any, ClassVar, TYPE_CHECKING

if TYPE_CHECKING:
    from pfeed.storages.storage_config import StorageConfig
    from pfeed._io.io_config import IOConfig
    from pfund.datas.data_config import DataConfig

from pfeed.enums import DataCategory, DataSource


class BaseData:
    category: ClassVar[DataCategory]

    def __init__(
        self,
        data_config: DataConfig,
        storage_config: StorageConfig,
        io_config: IOConfig,
    ):
        self.config: DataConfig = data_config
        self.storage_config: StorageConfig = storage_config
        self.io_config: IOConfig = io_config
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
            "data_config": self.config.model_dump(),
            "storage_config": self.storage_config.model_dump(),
            "io_config": self.io_config.model_dump(),
        }

    def update_extra_data(self, extra_data: dict[str, Any]):
        self.extra_data = extra_data

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, BaseData):
            return NotImplemented
        return self.source == other.source and self.origin == other.origin

    def __hash__(self):
        return hash((self.source, self.origin))
