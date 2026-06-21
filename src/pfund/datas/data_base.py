from __future__ import annotations

from typing import Any, ClassVar

from pfeed.enums import DataCategory, DataSource

from pfund.datas.data_config import DataConfig


class BaseData:
    category: ClassVar[DataCategory]

    def __init__(self, data_config: DataConfig | None = None):
        self.config: DataConfig = data_config or DataConfig()
        self.extra: dict[str, Any] = {}

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
        }

    def update_extra(self, extra: dict[str, Any]):
        self.extra = extra

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, BaseData):
            return NotImplemented
        return self.source == other.source and self.origin == other.origin

    def __hash__(self):
        return hash((self.source, self.origin))
