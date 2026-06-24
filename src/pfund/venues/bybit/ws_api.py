from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar, Literal

if TYPE_CHECKING:
    from collections.abc import Awaitable

    from pfund.brokers.crypto.exchanges.bybit.ws_api_base import BybitWebSocketAPI
    from pfund.venues._apis.ws_api_base import Message, WebSocketName
    from pfund.datas.resolution import Resolution
    from pfund.venues._crypto.account_api import CryptoAccount
    from pfund.enums import Environment
    from pfund.typing import FullDataChannel

import asyncio

from pfund.venues._apis.ws_api_base import BaseWebSocketAPI
from pfund.venues.bybit.product import BybitProduct
from pfund.enums import CryptoExchange

ProductCategory = BybitProduct.Category


class BybitWebSocketAPI(BaseWebSocketAPI):
    """A facade for the Bybit websocket API."""

    EXCHANGE: ClassVar[CryptoExchange] = CryptoExchange.BYBIT

    def __init__(self, env: Environment):
        super().__init__(env)
        self._apis: dict[ProductCategory, BybitWebSocketAPI] = {
            ProductCategory.LINEAR: self.get_api_class("linear")(env),
            ProductCategory.INVERSE: self.get_api_class("inverse")(env),
            ProductCategory.SPOT: self.get_api_class("spot")(env),
            ProductCategory.OPTION: self.get_api_class("option")(env),
        }

    @staticmethod
    def get_api_class(category: ProductCategory | str) -> type[BybitWebSocketAPI]:
        from pfund.venues.bybit._ws_apis.ws_api_inverse import (
            BybitInverseWebSocketAPI,
        )
        from pfund.venues.bybit._ws_apis.ws_api_linear import (
            BybitLinearWebSocketAPI,
        )
        from pfund.venues.bybit._ws_apis.ws_api_option import (
            BybitOptionWebSocketAPI,
        )
        from pfund.venues.bybit._ws_apis.ws_api_spot import (
            BybitSpotWebSocketAPI,
        )

        category = ProductCategory[category.upper()]
        return {
            ProductCategory.LINEAR: BybitLinearWebSocketAPI,
            ProductCategory.INVERSE: BybitInverseWebSocketAPI,
            ProductCategory.SPOT: BybitSpotWebSocketAPI,
            ProductCategory.OPTION: BybitOptionWebSocketAPI,
        }[category]

    def get_api(self, category: ProductCategory | str | None = None):
        # for some actions that are not specific to a product category, just use the first api
        # e.g. connecting to private channels
        if category is None:
            return list(self._apis.values())[0]
        else:
            return self._apis[ProductCategory[category.upper()]]

    async def _subscribe(self, *args: Any, **kwargs: Any):
        raise NotImplementedError(
            "this method should not be called in this Websocket Facade class"
        )

    async def _unsubscribe(self, *args: Any, **kwargs: Any):
        raise NotImplementedError(
            "this method should not be called in this Websocket Facade class"
        )

    async def _authenticate(self, *args: Any, **kwargs: Any):
        raise NotImplementedError(
            "this method should not be called in this Websocket Facade class"
        )

    async def _ping(self, *args: Any, **kwargs: Any):
        raise NotImplementedError(
            "this method should not be called in this Websocket Facade class"
        )

    async def _on_message(self, *args: Any, **kwargs: Any):
        raise NotImplementedError(
            "this method should not be called in this Websocket Facade class"
        )

    def _parse_message(self, msg: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError(
            "this method should not be called in this Websocket Facade class"
        )

    def _create_public_channel(
        self, product: BybitProduct, resolution: Resolution
    ) -> str:
        api = self.get_api(product.category)
        return api._create_public_channel(product, resolution)

    def set_logger(self, name: str):
        super().set_logger(name)
        for api in self._apis.values():
            api.set_logger(name)

    def set_callback(
        self,
        callback: Callable[[WebSocketName, Message], Awaitable[None] | None],
        raw_msg: bool = False,
    ):
        for api in self._apis.values():
            api.set_callback(callback, raw_msg=raw_msg)

    def add_account(self, account: CryptoAccount) -> CryptoAccount:
        api = self.get_api()
        return api.add_account(account)

    def add_product(self, product: BybitProduct) -> BybitProduct:
        api = self.get_api(product.category)
        return api.add_product(product)

    def add_channel(
        self,
        channel: FullDataChannel,
        *,
        channel_type: Literal["public", "private"],
        category: ProductCategory | str | None = None,
    ):
        api = self.get_api(category)
        api.add_channel(channel, channel_type=channel_type)

    async def connect(self):
        try:
            async with asyncio.TaskGroup() as task_group:
                for api in self._apis.values():
                    task_group.create_task(api.connect())
        except asyncio.CancelledError:
            self._logger.warning(f"{self.exch} connect() was cancelled")

    async def disconnect(self, reason: str = ""):
        async with asyncio.TaskGroup() as task_group:
            for api in self._apis.values():
                task_group.create_task(api.disconnect(reason=reason))
