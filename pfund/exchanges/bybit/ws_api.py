from __future__ import annotations
from typing import TYPE_CHECKING, Literal, Callable, Awaitable

from pfund.accounts.account_crypto import CryptoAccount
if TYPE_CHECKING:
    import logging
    from pfund.typing import tEnvironment, FullDataChannel
    from pfund.datas.resolution import Resolution
    from pfund.exchanges.bybit.exchange import tProductCategory
    from pfund.exchanges.bybit.ws_api_bybit import BybitWebsocketApi
    from pfund.enums import Environment

import asyncio

from pfund.enums import CryptoExchange
from pfund.products.product_bybit import BybitProduct
from pfund.exchanges.ws_api_base import BaseWebsocketApi

ProductCategory = BybitProduct.ProductCategory


class WebsocketApi(BaseWebsocketApi):
    '''A facade for the Bybit websocket API.'''
    exch = CryptoExchange.BYBIT
    
    def __init__(self, env: Environment | tEnvironment):
        from pfund.exchanges.bybit.ws_api_linear import WebsocketApiLinear
        from pfund.exchanges.bybit.ws_api_inverse import WebsocketApiInverse
        from pfund.exchanges.bybit.ws_api_spot import WebsocketApiSpot
        from pfund.exchanges.bybit.ws_api_option import WebsocketApiOption

        super().__init__(env)
        
        self._apis: dict[ProductCategory, BybitWebsocketApi] = {
            ProductCategory.LINEAR: WebsocketApiLinear(env),
            ProductCategory.INVERSE: WebsocketApiInverse(env),
            ProductCategory.SPOT: WebsocketApiSpot(env),
            ProductCategory.OPTION: WebsocketApiOption(env),
        }
        
    def get_api(self, category: tProductCategory | None=None):
        # for some actions that are not specific to a product category, just use the first api
        # e.g. connecting to private channels
        if category is None:
            return list(self._apis.values())[0]
        else:
            return self._apis[ProductCategory[category.upper()]]
    
    async def _subscribe(self, *args, **kwargs):
        raise NotImplementedError("this method should not be called in this Websocket Facade class")
    
    async def _unsubscribe(self, *args, **kwargs):
        raise NotImplementedError("this method should not be called in this Websocket Facade class")
    
    async def _authenticate(self, *args, **kwargs):
        raise NotImplementedError("this method should not be called in this Websocket Facade class")
    
    async def _ping(self, *args, **kwargs):
        raise NotImplementedError("this method should not be called in this Websocket Facade class")
    
    def _create_public_channel(self, product: BybitProduct, resolution: Resolution):
        api = self.get_api(product.category)
        return api._create_public_channel(product, resolution)
    
    def set_logger(self, logger: logging.Logger):
        super().set_logger(logger)
        for api in self._apis.values():
            api.set_logger(logger)
    
    def set_callback(self, callback: Callable[[str], Awaitable[None] | None]):
        for api in self._apis.values():
            api.set_callback(callback)
    
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
        channel_type: Literal['public', 'private'],
        category: ProductCategory | tProductCategory | None=None,
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
