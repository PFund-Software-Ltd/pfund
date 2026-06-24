from __future__ import annotations
from typing import TYPE_CHECKING, ClassVar, Any

if TYPE_CHECKING:
    from pfund.entities import BaseProduct, BaseAccount, BaseOrder
    from pfund.engines.settings.trade_engine_settings import TradeEngineSettings

from pfund.venues.exchange_crypto import CryptoExchange
from pfund.venues.hyperliquid.product import HyperliquidProduct
from pfund.venues.hyperliquid.account import HyperliquidAccount
from pfund.venues.hyperliquid.order import HyperliquidOrder
from pfund.enums import TradingVenue, Environment


class Hyperliquid(CryptoExchange):
    name: ClassVar[TradingVenue] = TradingVenue.HYPERLIQUID
    Product: ClassVar[type[BaseProduct]] = HyperliquidProduct
    Account: ClassVar[type[BaseAccount]] = HyperliquidAccount
    Order: ClassVar[type[BaseOrder]] = HyperliquidOrder

    def __init__(
        self, env: Environment | str, settings: TradeEngineSettings | None = None
    ):
        from pfund.venues.hyperliquid.rest_api import HyperliquidRESTfulAPI
        from pfund.venues.hyperliquid.ws_api import HyperliquidWebSocketAPI

        super().__init__(env=env, settings=settings)
        self.rest_api = HyperliquidRESTfulAPI(env=self._env)
        self.ws_api = HyperliquidWebSocketAPI(env=self._env)
