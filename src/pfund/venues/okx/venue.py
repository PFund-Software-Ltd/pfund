from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal

if TYPE_CHECKING:
    from pfund.engines.settings.trade_engine_settings import TradeEngineSettings

from pfund.datas.timeframe import Timeframe
from pfund.venues.crypto_exchange import CryptoExchange
from pfund.venues.okx.rest_api import OKXRestAPI
from pfund.venues.okx.ws_api import OKXWebSocketAPI
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
    CryptoExchange[
        OKXRestAPI,
        OKXWebSocketAPI,
        OKXConfig,
        OKXMarket,
        OKXAccount,
        OKXProduct,
        OKXOrder,
        OKXBalance,
        OKXBalance.Snapshot,
        OKXPosition,
        OKXPosition.Snapshot,
    ]
):
    name: ClassVar[TradingVenue] = TradingVenue.OKX
    adapter: ClassVar[OKXAdapter] = OKXAdapter()

    RestAPI: ClassVar[type[OKXRestAPI]] = OKXRestAPI
    WSAPI: ClassVar[type[OKXWebSocketAPI]] = OKXWebSocketAPI

    Config: ClassVar[type[OKXConfig]] = OKXConfig
    Market: ClassVar[type[OKXMarket]] = OKXMarket
    Account: ClassVar[type[OKXAccount]] = OKXAccount
    Balance: ClassVar[type[OKXBalance]] = OKXBalance
    Order: ClassVar[type[OKXOrder]] = OKXOrder
    Product: ClassVar[type[OKXProduct]] = OKXProduct
    Position: ClassVar[type[OKXPosition]] = OKXPosition

    METADATA: ClassVar[VenueMetadata] = VenueMetadata()
