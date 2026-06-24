from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from pfund.entities import BaseProduct, BaseAccount, BaseOrder
    from pfund.engines.settings.trade_engine_settings import TradeEngineSettings

from pfund.enums import TradingVenue, Environment
from pfund.venues.okx.product import OKXProduct
from pfund.venues.okx.account import OKXAccount
from pfund.venues.okx.order import OKXOrder
from pfund.venues.exchange_crypto import CryptoExchange


class OKX(CryptoExchange):
    name: ClassVar[TradingVenue] = TradingVenue.OKX
    Product: ClassVar[type[BaseProduct]] = OKXProduct
    Account: ClassVar[type[BaseAccount]] = OKXAccount
    Order: ClassVar[type[BaseOrder]] = OKXOrder

    def __init__(
        self, env: Environment | str, settings: TradeEngineSettings | None = None
    ):
        from pfund.venues.okx.rest_api import OKXRESTfulAPI
        from pfund.venues.okx.ws_api import OKXWebSocketAPI

        super().__init__(env=env, settings=settings)
        self.rest_api = OKXRESTfulAPI(env=self._env)
        self.ws_api = OKXWebSocketAPI(env=self._env)
