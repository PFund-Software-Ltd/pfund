from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pfund.entities.products.product_base import BaseProduct
    from pfund.datas.timeframe import Timeframe


def get_supported_resolutions(product: BaseProduct) -> dict[Timeframe, list[int]]:
    import importlib
    from pfund.enums import Broker

    supported_resolutions: dict[Timeframe, list[int]]
    broker = product.broker
    if broker == Broker.CRYPTO:
        Exchange = getattr(importlib.import_module(f'pfund.brokers.crypto.exchanges.{product.exchange.lower()}.exchange'), 'Exchange')
        supported_resolutions = Exchange.get_supported_resolutions(product)
    elif broker == Broker.IBKR:
        InteractiveBrokersAPI = getattr(importlib.import_module('pfund.brokers.ibkr.api'), 'InteractiveBrokersAPI')
        supported_resolutions = InteractiveBrokersAPI.SUPPORTED_RESOLUTIONS
    else:
        raise NotImplementedError(f'broker {broker} is not supported')
    return supported_resolutions