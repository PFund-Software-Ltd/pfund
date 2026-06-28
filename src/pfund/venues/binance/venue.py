from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal

if TYPE_CHECKING:
    from pfund.engines.settings.trade_engine_settings import TradeEngineSettings

from pfund.datas.timeframe import Timeframe
from pfund.venues.venue_base import BaseVenue
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
    BaseVenue[
        BinanceConfig,
        BinanceMarket,
        BinanceAccount,
        BinanceBalance,
        BinanceOrder,
        BinanceProduct,
        BinancePosition,
    ]
):
    name: ClassVar[TradingVenue] = TradingVenue.BINANCE
    adapter: ClassVar[BinanceAdapter] = BinanceAdapter()
    Config: ClassVar[type[BinanceConfig]] = BinanceConfig
    Market: ClassVar[type[BinanceMarket]] = BinanceMarket
    Order: ClassVar[type[BinanceOrder]] = BinanceOrder
    Product: ClassVar[type[BinanceProduct]] = BinanceProduct

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
        config: BinanceConfig | None = None,
        settings: TradeEngineSettings | None = None,
    ):
        from pfund.venues.binance.rest_api import BinanceRestAPI
        from pfund.venues.binance.ws_api import BinanceWebSocketAPI

        super().__init__(env=env, config=config, settings=settings)
        self.rest_api = BinanceRestAPI(env=self._env)
        self.ws_api = BinanceWebSocketAPI(env=self._env)
