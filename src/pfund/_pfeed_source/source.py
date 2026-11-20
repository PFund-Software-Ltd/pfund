from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal

if TYPE_CHECKING:
    from enum import StrEnum

    from pfeed.base.feed import BaseFeed

from pfeed.enums import DataAccessType, DataProviderType
from pfeed.source import BaseSource, SourceMetadata

from pfund._pfeed_source.component.feed import PFundComponentFeed
from pfund._pfeed_source.data_category import PFundDataCategory
from pfund._pfeed_source.engine.feed import PFundEngineFeed


class PFund[PipelineT: bool = Literal[False]](BaseSource[PipelineT]):
    Feeds: ClassVar[dict[StrEnum, type[BaseFeed]]] = {
        PFundDataCategory.COMPONENT_DATA: PFundComponentFeed,
        PFundDataCategory.ENGINE_DATA: PFundEngineFeed,
    }
    metadata: ClassVar[SourceMetadata] = SourceMetadata(
        name="PFUND",
        url="https://pfund.ai",
        # pfund's data is the engine's own output, it has no data types or asset types like market data
        data_categories={
            PFundDataCategory.COMPONENT_DATA: {},
            PFundDataCategory.ENGINE_DATA: {},
        },
        # the data is produced by pfund itself, like a venue's own data
        provider_type=DataProviderType.VENUE,
        access_type=DataAccessType.FREE,
    )

    component_feed: PFundComponentFeed[PipelineT]
    engine_feed: PFundEngineFeed[PipelineT]
