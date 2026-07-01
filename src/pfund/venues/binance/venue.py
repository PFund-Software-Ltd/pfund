from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal

if TYPE_CHECKING:
    from pfund.engines.settings.trade_engine_settings import TradeEngineSettings

from pfund.datas.timeframe import Timeframe
from pfund.venues.crypto_exchange import CryptoExchange
from pfund.venues.binance.rest_api import BinanceRestAPI
from pfund.venues.binance.ws_api import BinanceWebSocketAPI
from pfund.venues.venue_metadata import VenueMetadata
from pfund.venues.binance.adapter import BinanceAdapter
from pfund.venues.binance.config import BinanceConfig
from pfund.venues.binance.market import BinanceMarket
from pfund.venues.binance.account import BinanceAccount
from pfund.venues.binance.balance import BinanceBalance
from pfund.venues.binance.order import BinanceOrder
from pfund.venues.binance.product import BinanceProduct
from pfund.venues.binance.position import BinancePosition
from pfund.enums import TradingVenue, Environment, AssetTypeModifier, CryptoAssetType


class Binance(
    CryptoExchange[
        BinanceRestAPI,
        BinanceWebSocketAPI,
        BinanceConfig,
        BinanceMarket,
        BinanceAccount,
        BinanceProduct,
        BinanceOrder,
        BinanceBalance,
        BinanceBalance.Snapshot,
        BinancePosition,
        BinancePosition.Snapshot,
    ]
):
    name: ClassVar[TradingVenue] = TradingVenue.BINANCE
    adapter: ClassVar[BinanceAdapter] = BinanceAdapter()

    RestAPI: ClassVar[type[BinanceRestAPI]] = BinanceRestAPI
    WebSocketAPI: ClassVar[type[BinanceWebSocketAPI]] = BinanceWebSocketAPI

    Config: ClassVar[type[BinanceConfig]] = BinanceConfig
    Market: ClassVar[type[BinanceMarket]] = BinanceMarket
    Account: ClassVar[type[BinanceAccount]] = BinanceAccount
    Balance: ClassVar[type[BinanceBalance]] = BinanceBalance
    Order: ClassVar[type[BinanceOrder]] = BinanceOrder
    Product: ClassVar[type[BinanceProduct]] = BinanceProduct
    Position: ClassVar[type[BinancePosition]] = BinancePosition

    METADATA: ClassVar[VenueMetadata] = VenueMetadata(
        requires_asyncio_loop=True,
    )
