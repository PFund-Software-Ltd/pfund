from typing import ClassVar

from pfeed.enums import DataCategory
from pfeed.source import BaseSource


class PFund(BaseSource):
    name: ClassVar[str] = "PFUND"

    def get_data_categories(self) -> list[DataCategory]:
        return [DataCategory.ENGINE_DATA, DataCategory.COMPONENT_DATA]
