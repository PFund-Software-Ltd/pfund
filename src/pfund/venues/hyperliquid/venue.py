from __future__ import annotations
from typing import TYPE_CHECKING, ClassVar, Any, Literal

if TYPE_CHECKING:
    from pfund.engines.settings.trade_engine_settings import TradeEngineSettings

from pfund.datas.timeframe import Timeframe
from pfund.venues.venue_base import BaseVenue
from pfund.venues.venue_metadata import VenueMetadata
from pfund.venues.hyperliquid.adapter import HyperliquidAdapter
from pfund.venues.hyperliquid.config import HyperliquidConfig
from pfund.venues.hyperliquid.market import HyperliquidMarket
from pfund.venues.hyperliquid.account import HyperliquidAccount
from pfund.venues.hyperliquid.balance import HyperliquidBalance
from pfund.venues.hyperliquid.order import HyperliquidOrder
from pfund.venues.hyperliquid.product import HyperliquidProduct
from pfund.venues.hyperliquid.position import HyperliquidPosition
from pfund.enums import TradingVenue, Environment, AssetTypeModifier, CryptoAssetType


class Hyperliquid(
    BaseVenue[
        HyperliquidConfig,
        HyperliquidMarket,
        HyperliquidAccount,
        HyperliquidBalance,
        HyperliquidOrder,
        HyperliquidProduct,
        HyperliquidPosition,
    ]
):
    name: ClassVar[TradingVenue] = TradingVenue.HYPERLIQUID
    adapter: ClassVar[HyperliquidAdapter] = HyperliquidAdapter()
    Config: ClassVar[type[HyperliquidConfig]] = HyperliquidConfig
    Market: ClassVar[type[HyperliquidMarket]] = HyperliquidMarket
    Order: ClassVar[type[HyperliquidOrder]] = HyperliquidOrder
    Product: ClassVar[type[HyperliquidProduct]] = HyperliquidProduct

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
        config: HyperliquidConfig | None = None,
        settings: TradeEngineSettings | None = None,
    ):
        from pfund.venues.hyperliquid.rest_api import HyperliquidRESTfulAPI
        from pfund.venues.hyperliquid.ws_api import HyperliquidWebSocketAPI

        super().__init__(env=env, config=config, settings=settings)
        self.rest_api = HyperliquidRESTfulAPI(env=self._env)
        self.ws_api = HyperliquidWebSocketAPI(env=self._env)
