# FIXME: this is a temporary enum for database, to be removed when pfund supports all databases from pfeed
from enum import StrEnum
from pfeed.enums.data_storage import DatabaseDataStorage as PFeedDatabase


class Database(StrEnum):
    DUCKDB = PFeedDatabase.DUCKDB