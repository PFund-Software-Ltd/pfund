from enum import StrEnum
from pfeed.enums.data_storage import DataStorage


class Database(StrEnum):
    DUCKDB = DataStorage.DUCKDB