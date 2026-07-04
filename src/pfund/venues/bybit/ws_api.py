# pyright: reportUnknownParameterType=false
from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from pfund.venues._apis.typing import ResponseData
    from pfund.venues._apis.ws_api_base import RawMessage, WebSocketName
    from pfund.venues.bybit._ws_apis.ws_api_base import BybitBaseWebSocketAPI
    from pfund.datas.resolution import Resolution
    from pfund.enums import Environment
    from pfund.typing import FullDataChannel

import asyncio

from pfund.venues._apis.ws_api_base import BaseWebSocketAPI
from pfund.venues.bybit.config import BybitConfig
from pfund.venues.bybit.product import BybitProduct
from pfund.venues.bybit.account import BybitAccount
from pfund.venues.bybit._ws_apis import (
    BybitInverseWebSocketAPI,
    BybitLinearWebSocketAPI,
    BybitOptionWebSocketAPI,
    BybitSpotWebSocketAPI,
)
from pfund.enums import TradingVenue


class BybitWebSocketAPI(BaseWebSocketAPI[BybitConfig, BybitAccount, BybitProduct]):
    """A facade for the Bybit websocket API."""

    venue: ClassVar[TradingVenue] = TradingVenue.BYBIT
    APIS: ClassVar[dict[BybitProduct.Category, type[BybitBaseWebSocketAPI]]] = {
        BybitProduct.Category.LINEAR: BybitLinearWebSocketAPI,
        BybitProduct.Category.INVERSE: BybitInverseWebSocketAPI,
        BybitProduct.Category.SPOT: BybitSpotWebSocketAPI,
        BybitProduct.Category.OPTION: BybitOptionWebSocketAPI,
    }

    def __init__(
        self,
        env: Literal[Environment.PAPER, Environment.LIVE],
        data_mode: bool = False,
    ):
        super().__init__(env=env, data_mode=data_mode)
        self._apis: dict[BybitProduct.Category, BybitBaseWebSocketAPI] = {
            category: API(env=env, data_mode=data_mode)
            for category, API in self.APIS.items()
        }

    def get_api(self, category: BybitProduct.Category | str | None = None):
        # for some actions that are not specific to a product category, just use the first api
        # e.g. connecting to private channels
        if category is None:
            return list(self._apis.values())[0]
        else:
            return self._apis[BybitProduct.Category[category.upper()]]

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

    def _create_market_data_channel(
        self, product: BybitProduct, resolution: Resolution
    ) -> str:
        api = self.get_api(product.category)
        return api._create_market_data_channel(product, resolution)

    def set_callback(
        self,
        callback: Callable[
            [WebSocketName, RawMessage | ResponseData], Awaitable[None] | None
        ],
        raw_msg: bool = False,
    ):
        for api in self._apis.values():
            api.set_callback(callback, raw_msg=raw_msg)

    def add_account(self, account: BybitAccount) -> None:
        api = self.get_api()
        return api.add_account(account)

    def add_product(self, product: BybitProduct) -> None:
        api = self.get_api(product.category)
        return api.add_product(product)

    def add_channel(
        self,
        channel: FullDataChannel,
        *,
        channel_type: Literal["public", "private"],
        category: BybitProduct.Category | str | None = None,
    ):
        api = self.get_api(category)
        api.add_channel(channel, channel_type=channel_type)

    async def _connect(self):
        async with asyncio.TaskGroup() as task_group:
            for api in self._apis.values():
                task_group.create_task(api._connect())

    async def disconnect(self, reason: str = ""):
        async with asyncio.TaskGroup() as task_group:
            for api in self._apis.values():
                task_group.create_task(api.disconnect(reason=reason))
