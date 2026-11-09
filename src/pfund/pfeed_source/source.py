from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from pfeed.base.feed import BaseFeed

from pfeed.enums import DataAccessType, DataCategory, DataProviderType
from pfeed.source import BaseSource, SourceMetadata

from pfeed_pfund.component.feed import PFundComponentFeed
from pfeed_pfund.engine.feed import PFundEngineFeed


class PFund(BaseSource):
    Feeds: ClassVar[dict[DataCategory, type[BaseFeed]]] = {
        DataCategory.COMPONENT_DATA: PFundComponentFeed,
        DataCategory.ENGINE_DATA: PFundEngineFeed,
    }
    metadata: ClassVar[SourceMetadata] = SourceMetadata(
        name="PFUND",
        url="https://pfund.ai",
        # pfund's data is the engine's own output, it has no data types or asset types like market data
        data_categories={
            DataCategory.COMPONENT_DATA: {},
            DataCategory.ENGINE_DATA: {},
        },
        # the data is produced by pfund itself, like a venue's own data
        provider_type=DataProviderType.VENUE,
        access_type=DataAccessType.FREE,
    )

    component_feed: PFundComponentFeed
    engine_feed: PFundEngineFeed
