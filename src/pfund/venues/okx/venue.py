from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal

if TYPE_CHECKING:
    from pfund.engines.settings.trade_engine_settings import TradeEngineSettings

from pfund.datas.timeframe import Timeframe
from pfund.venues.venue_base import BaseVenue
from pfund.venues.venue_metadata import VenueMetadata
from pfund.venues.okx.adapter import OKXAdapter
from pfund.venues.okx.config import OKXConfig
from pfund.venues.okx.market import OKXMarket
from pfund.venues.okx.account import OKXAccount
from pfund.venues.okx.balance import OKXBalance
from pfund.venues.okx.order import OKXOrder
from pfund.venues.okx.product import OKXProduct
from pfund.venues.okx.position import OKXPosition
from pfund.enums import TradingVenue, Environment, AssetTypeModifier, CryptoAssetType


class OKX(
    BaseVenue[
        OKXConfig, OKXMarket, OKXAccount, OKXBalance, OKXOrder, OKXProduct, OKXPosition
    ]
):
    name: ClassVar[TradingVenue] = TradingVenue.OKX
    adapter: ClassVar[OKXAdapter] = OKXAdapter()
    Config: ClassVar[type[OKXConfig]] = OKXConfig
    Market: ClassVar[type[OKXMarket]] = OKXMarket
    Order: ClassVar[type[OKXOrder]] = OKXOrder
    Product: ClassVar[type[OKXProduct]] = OKXProduct

    METADATA: ClassVar[VenueMetadata] = VenueMetadata()

    def __init__(
        self,
        env: Literal[
            Environment.SANDBOX,
            Environment.PAPER,
            Environment.LIVE,
            "SANDBOX",
            "PAPER",
            "LIVE",
        ],
        config: OKXConfig | None = None,
        settings: TradeEngineSettings | None = None,
    ):
        from pfund.venues.okx.rest_api import OKXRestAPI
        from pfund.venues.okx.ws_api import OKXWebSocketAPI

        super().__init__(env=env, config=config, settings=settings)
        self.rest_api = OKXRestAPI(env=self._env)
        self.ws_api = OKXWebSocketAPI(env=self._env)
