from __future__ import annotations
from typing import TYPE_CHECKING, Literal, Callable
if TYPE_CHECKING:
    import logging
    from pfund.typing import tEnvironment, FullDataChannel
    from pfund.datas.resolution import Resolution
    from pfund.exchanges.bybit.exchange import tProductCategory
    from pfund.exchanges.bybit.ws_api_bybit import BybitWebsocketApi
    from pfund.enums import Environment, PrivateDataChannel

import asyncio

from pfund.enums import CryptoExchange
from pfund.products.product_bybit import BybitProduct
from pfund.exchanges.ws_api_base import BaseWebsocketApi

ProductCategory = BybitProduct.ProductCategory


# FIXME: inherit from BaseWebsocketApi when the functions are implemented
# class WebsocketApi(BaseWebsocketApi):
class WebsocketApi:
    '''A facade for the Bybit websocket API.'''
    name = CryptoExchange.BYBIT
    
    def __init__(self, env: Environment | tEnvironment):
        from pfund.exchanges.bybit.ws_api_linear import WebsocketApiLinear
        from pfund.exchanges.bybit.ws_api_inverse import WebsocketApiInverse
        from pfund.exchanges.bybit.ws_api_spot import WebsocketApiSpot
        from pfund.exchanges.bybit.ws_api_option import WebsocketApiOption
        
        self._apis: dict[ProductCategory, BybitWebsocketApi] = {
            ProductCategory.LINEAR: WebsocketApiLinear(env),
            ProductCategory.INVERSE: WebsocketApiInverse(env),
            ProductCategory.SPOT: WebsocketApiSpot(env),
            ProductCategory.OPTION: WebsocketApiOption(env),
        }
        # TODO: create self._servers?
    
    def get_api(self, category: tProductCategory | None=None):
        # for some actions that are not specific to a product category, just use the first api
        # e.g. connecting to private channels
        if category is None:
            return list(self._apis.values())[0]
        else:
            return self._apis[ProductCategory[category.upper()]]
        
    def set_logger(self, logger: logging.Logger):
        for api in self._apis.values():
            api.set_logger(logger)
    
    def set_callback(self, callback: Callable[[str], None]):
        for api in self._apis.values():
            api.set_callback(callback)
    
    async def connect(self):
        async with asyncio.TaskGroup() as task_group:
            for api in self._apis.values():
                task_group.create_task(api.connect()) 
    
    async def disconnect(self, reason: str = ""):
        async with asyncio.TaskGroup() as task_group:
            for api in self._apis.values():
                task_group.create_task(api.disconnect(reason=reason))

    def _create_public_channel(self, product: BybitProduct, resolution: Resolution):
        api = self.get_api(product.category)
        return api._create_public_channel(product, resolution)
    
    def add_channel(
        self, 
        channel: PrivateDataChannel | FullDataChannel, 
        *,
        channel_type: Literal['public', 'private'],
        category: ProductCategory | tProductCategory | None=None,
    ):
        api = self.get_api(category)
        api.add_channel(channel, channel_type)