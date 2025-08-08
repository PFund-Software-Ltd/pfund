from __future__ import annotations
from typing import TYPE_CHECKING, Literal, Callable, Awaitable

from pfund.accounts.account_crypto import CryptoAccount
if TYPE_CHECKING:
    import logging
    from pfund._typing import tEnvironment, FullDataChannel
    from pfund.datas.resolution import Resolution
    from pfund.exchanges.bybit.exchange import tProductCategory
    from pfund.exchanges.bybit.ws_api_bybit import BybitWebSocketAPI
    from pfund.enums import Environment

import asyncio

from pfund.enums import CryptoExchange
from pfund.products.product_bybit import BybitProduct
from pfund.exchanges.ws_api_base import BaseWebSocketAPI

ProductCategory = BybitProduct.ProductCategory


class WebSocketAPI(BaseWebSocketAPI):
    '''A facade for the Bybit websocket API.'''
    exch = CryptoExchange.BYBIT
    
    def __init__(self, env: Environment | tEnvironment):
        super().__init__(env)
        self._apis: dict[ProductCategory, BybitWebSocketAPI] = {
            ProductCategory.LINEAR: self._get_api_class('linear')(env),
            ProductCategory.INVERSE: self._get_api_class('inverse')(env),
            ProductCategory.SPOT: self._get_api_class('spot')(env),
            ProductCategory.OPTION: self._get_api_class('option')(env),
        }
    
    @staticmethod
    def _get_api_class(category: ProductCategory | tProductCategory) -> type[BybitWebSocketAPI]:
        from pfund.exchanges.bybit.ws_api_linear import LinearWebSocketAPI
        from pfund.exchanges.bybit.ws_api_inverse import InverseWebSocketAPI
        from pfund.exchanges.bybit.ws_api_spot import SpotWebSocketAPI
        from pfund.exchanges.bybit.ws_api_option import OptionWebSocketAPI
        category = ProductCategory[category.upper()]
        return {
            ProductCategory.LINEAR: LinearWebSocketAPI,
            ProductCategory.INVERSE: InverseWebSocketAPI,
            ProductCategory.SPOT: SpotWebSocketAPI,
            ProductCategory.OPTION: OptionWebSocketAPI,
        }[category]
        
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
    
    async def _on_message(self, *args, **kwargs):
        raise NotImplementedError("this method should not be called in this Websocket Facade class")
    
    def _parse_message(self, msg: dict) -> dict:
        raise NotImplementedError("this method should not be called in this Websocket Facade class")
    
    def _create_public_channel(self, product: BybitProduct, resolution: Resolution):
        api = self.get_api(product.category)
        return api._create_public_channel(product, resolution)
    
    def set_logger(self, name: str):
        super().set_logger(name)
        for api in self._apis.values():
            api.set_logger(name)
    
    def set_callback(self, callback: Callable[[str], Awaitable[None] | None], raw_msg: bool=False):
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
