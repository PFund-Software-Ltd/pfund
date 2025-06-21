from enum import StrEnum


class Database(StrEnum):
    DUCKDB = 'DUCKDB'
    POSTGRESQL = 'POSTGRESQL'