from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from pfund.entities import BaseProduct, BaseAccount, BaseOrder
    from pfund.engines.settings.trade_engine_settings import TradeEngineSettings

from pfund.enums import TradingVenue, Environment
from pfund.venues.binance.product import BinanceProduct
from pfund.venues.binance.account import BinanceAccount
from pfund.venues.binance.order import BinanceOrder
from pfund.venues.exchange_crypto import CryptoExchange


class Binance(CryptoExchange):
    name: ClassVar[TradingVenue] = TradingVenue.BINANCE
    Product: ClassVar[type[BaseProduct]] = BinanceProduct
    Account: ClassVar[type[BaseAccount]] = BinanceAccount
    Order: ClassVar[type[BaseOrder]] = BinanceOrder

    def __init__(
        self, env: Environment | str, settings: TradeEngineSettings | None = None
    ):
        from pfund.venues.binance.rest_api import BinanceRESTfulAPI
        from pfund.venues.binance.ws_api import BinanceWebSocketAPI

        super().__init__(env=env, settings=settings)
        self.rest_api = BinanceRESTfulAPI(env=self._env)
        self.ws_api = BinanceWebSocketAPI(env=self._env)
