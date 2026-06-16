from enum import StrEnum

from pfeed.enums.io_format import IOFormat


class DataLake(StrEnum):
    DELTALAKE = IOFormat.DELTALAKE
