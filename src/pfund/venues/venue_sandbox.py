# pyright: reportUninitializedInstanceVariable=false, reportUnknownMemberType=false, reportAttributeAccessIssue=false, reportArgumentType=false, reportGeneralTypeIssues=false
from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Any
from typing_extensions import override

if TYPE_CHECKING:
    from pfeed.storages.storage_config import StorageConfig
    from pfeed.sources.pfund.engine_feed import PFundEngineFeed
    from pfund.typing import FullDataChannel
    from pfund.datas.data_market import MarketData
    from pfund.venues.venue_base import AnyVenue
    from pfund.venues._apis.typing import Result, ResponseData

import queue
import logging
from decimal import Decimal

from pfund.enums import Environment, TradingVenue
from pfund.venues.venue_base import BaseVenue
from pfund.venues.venue_config import VenueConfig
from pfund.entities import (
    BaseMarket,
    BaseAccount,
    BaseProduct,
    BaseOrder,
    BaseBalance,
    BasePosition,
)


class SandboxVenue(
    BaseVenue[
        VenueConfig,
        BaseMarket,
        BaseAccount,
        BaseProduct,
        BaseOrder,
        BaseBalance,
        BaseBalance.Snapshot,
        BasePosition,
        BasePosition.Snapshot,
    ]
):
    def __init__(
        self,
        venue: TradingVenue | str,
        engine_feed: PFundEngineFeed,
        storage_config: StorageConfig,
        replay_mode: bool = True,
        config: VenueConfig | None = None,
    ):
        self._venue = TradingVenue[venue.upper()]
        self._engine_feed = engine_feed
        self._storage_config = storage_config
        self._replay_mode = replay_mode
        VenueClass = self._VenueClass
        # Adopt the real venue's class-level config/metadata/adapter so BaseVenue's
        # __init__ and add_product operate on the REAL venue's markets and symbol
        # mapping. The venue-specific entity classes (Balance/Order/...) are reached
        # at runtime via self._VenueClass; the generic slots are just base classes.
        self.Config = VenueClass.Config
        self.METADATA = VenueClass.METADATA
        self.adapter = VenueClass.adapter
        # NOTE: __real_venue is ONLY used for receiving real live data (e.g. IBKR,
        # whose market data needs an authenticated connection). It is read-only so it
        # can never send orders, and in replay mode it is not constructed at all — no
        # credentials loaded. Built BEFORE super().__init__() so any init-time access
        # (and later connect/disconnect) is safe.
        self.__real_venue: AnyVenue | None = (
            None
            if replay_mode
            else VenueClass(
                env=Environment.LIVE,
                config=config,
                read_only=True,  # read-only MUST be true to avoid accidentally sending out orders etc.
            )
        )
        super().__init__(env=Environment.SANDBOX, config=config, read_only=False)

    @property
    def name(self) -> TradingVenue:
        return self._venue

    @property
    def _VenueClass(self) -> type[AnyVenue]:
        return self._venue.venue_class

    @override
    def refetch_markets(self) -> None:
        pass

    @override
    def add_channel(
        self,
        channel: FullDataChannel,
        *,
        channel_type: Literal["public", "private"] = "public",
    ) -> None:
        pass

    @override
    def _add_market_data_channel(self, data: MarketData) -> None:
        pass

    @override
    def _add_private_channels(self) -> None:
        pass

    @override
    def _build_balance_update(self, ts, data, account_name, source):
        return self._VenueClass._build_balance_update(ts, data, account_name, source)

    def connect(self):
        if self.__real_venue:
            self.__real_venue.connect()

    def disconnect(self, reason: str = ""):
        if self.__real_venue:
            self.__real_venue.disconnect(reason=reason)

    # TODO
    async def _get_balances(self, account: BaseAccount) -> Result:
        pass

    # TODO
    async def place_orders(self):
        pass
