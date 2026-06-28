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
from pfund.venues.bybit.product import BybitProduct
from pfund.venues.bybit.account import BybitAccount
from pfund.enums import TradingVenue


class BybitWebSocketAPI(BaseWebSocketAPI[BybitAccount, BybitProduct]):
    """A facade for the Bybit websocket API."""

    venue: ClassVar[TradingVenue] = TradingVenue.BYBIT

    def __init__(
        self,
        env: Literal[Environment.PAPER, Environment.LIVE],
        data_mode: bool = False,
    ):
        super().__init__(env=env, data_mode=data_mode)
        self._apis: dict[BybitProduct.Category, BybitBaseWebSocketAPI] = {
            BybitProduct.Category.LINEAR: self.get_api_class("linear")(
                env=env, data_mode=data_mode
            ),
            BybitProduct.Category.INVERSE: self.get_api_class("inverse")(
                env=env, data_mode=data_mode
            ),
            BybitProduct.Category.SPOT: self.get_api_class("spot")(
                env=env, data_mode=data_mode
            ),
            BybitProduct.Category.OPTION: self.get_api_class("option")(
                env=env, data_mode=data_mode
            ),
        }

    @staticmethod
    def get_api_class(
        category: BybitProduct.Category | str,
    ) -> type[BybitBaseWebSocketAPI]:
        from pfund.venues.bybit._ws_apis import (
            BybitInverseWebSocketAPI,
            BybitLinearWebSocketAPI,
            BybitOptionWebSocketAPI,
            BybitSpotWebSocketAPI,
        )

        category = BybitProduct.Category[category.upper()]
        return {
            BybitProduct.Category.LINEAR: BybitLinearWebSocketAPI,
            BybitProduct.Category.INVERSE: BybitInverseWebSocketAPI,
            BybitProduct.Category.SPOT: BybitSpotWebSocketAPI,
            BybitProduct.Category.OPTION: BybitOptionWebSocketAPI,
        }[category]

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

    def _add_account(self, account: BybitAccount) -> None:
        api = self.get_api()
        return api._add_account(account)

    def _add_product(self, product: BybitProduct) -> None:
        api = self.get_api(product.category)
        return api._add_product(product)

    def add_channel(
        self,
        channel: FullDataChannel,
        *,
        channel_type: Literal["public", "private"],
        category: BybitProduct.Category | str | None = None,
    ):
        api = self.get_api(category)
        api.add_channel(channel, channel_type=channel_type)

    async def connect(self):
        try:
            async with asyncio.TaskGroup() as task_group:
                for api in self._apis.values():
                    task_group.create_task(api.connect())
        except* asyncio.CancelledError:
            # re-raise so cooperative cancellation propagates to the caller
            self._logger.warning(f"{self.venue} connect() was cancelled")
            raise
        except* Exception:
            self._logger.exception(f"{self.venue} connect() failed")

    async def disconnect(self, reason: str = ""):
        async with asyncio.TaskGroup() as task_group:
            for api in self._apis.values():
                task_group.create_task(api.disconnect(reason=reason))
