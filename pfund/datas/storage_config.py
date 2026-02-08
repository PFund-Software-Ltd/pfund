from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator

from pfeed.enums import DataStorage, IOFormat, Compression, DataLayer


class StorageConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    data_path: Path | str | None = None
    data_layer: DataLayer = DataLayer.CLEANED
    data_domain: str = ''
    storage: DataStorage = DataStorage.LOCAL
    io_format: IOFormat = IOFormat.PARQUET
    compression: Compression = Compression.SNAPPY

    @field_validator('data_path', mode='before')
    @classmethod
    def validate_data_path(cls, data_path: Path | str | None) -> Path:
        if data_path is None:
            from pfeed import get_config
            pfeed_config = get_config()
            return pfeed_config.data_path
        return Path(data_path)