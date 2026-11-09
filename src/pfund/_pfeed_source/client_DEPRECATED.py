from typing import ClassVar

from pfeed.client import DataClient
from pfeed.enums import DataCategory
from pfeed.sources.pfund.component_feed import PFundComponentFeed
from pfeed.sources.pfund.engine_feed import PFundEngineFeed
from pfeed.sources.pfund.source import PFundSource


class PFund(DataClient):
    DataSource: ClassVar[type[PFundSource]] = PFundSource
    data_source: PFundSource

    engine_feed: PFundEngineFeed
    component_feed: PFundComponentFeed

    def _create_feeds(self):
        self.engine_feed = PFundEngineFeed(
            pipeline_mode=self._pipeline_mode,
            num_workers=(
                self._num_workers.get(DataCategory.ENGINE_DATA, None)
                if isinstance(self._num_workers, dict)
                else self._num_workers
            ),
        )
        self.component_feed = PFundComponentFeed(
            pipeline_mode=self._pipeline_mode,
            num_workers=(
                self._num_workers.get(DataCategory.COMPONENT_DATA, None)
                if isinstance(self._num_workers, dict)
                else self._num_workers
            ),
        )
