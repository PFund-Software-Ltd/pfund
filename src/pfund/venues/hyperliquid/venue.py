from __future__ import annotations
from typing import TYPE_CHECKING, ClassVar, Any, Literal

if TYPE_CHECKING:
    from pfund.engines.settings.trade_engine_settings import TradeEngineSettings

from pfund.datas.timeframe import Timeframe
from pfund.venues.crypto_exchange import CryptoExchange
from pfund.venues.hyperliquid.rest_api import HyperliquidRestAPI
from pfund.venues.hyperliquid.ws_api import HyperliquidWebSocketAPI
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
    CryptoExchange[
        HyperliquidRestAPI,
        HyperliquidWebSocketAPI,
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

    RestAPI: ClassVar[type[HyperliquidRestAPI]] = HyperliquidRestAPI
    WebSocketAPI: ClassVar[type[HyperliquidWebSocketAPI]] = HyperliquidWebSocketAPI

    Config: ClassVar[type[HyperliquidConfig]] = HyperliquidConfig
    Market: ClassVar[type[HyperliquidMarket]] = HyperliquidMarket
    Order: ClassVar[type[HyperliquidOrder]] = HyperliquidOrder
    Account: ClassVar[type[HyperliquidAccount]] = HyperliquidAccount
    Product: ClassVar[type[HyperliquidProduct]] = HyperliquidProduct

    METADATA: ClassVar[VenueMetadata] = VenueMetadata(
        requires_asyncio_loop=True,
    )
