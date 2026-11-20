from enum import StrEnum


class PFundDataCategory(StrEnum):
    """Data categories of pfund's data source, i.e. the keys of PFund.Feeds."""

    ENGINE_DATA = "ENGINE_DATA"
    COMPONENT_DATA = "COMPONENT_DATA"

    @property
    def feed_name(self) -> str:
        # e.g. PFundDataCategory.COMPONENT_DATA -> component_feed
        return self.lower().replace("_data", "_feed")
