from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pfund.entities.products.product_base import BaseProduct
    from pfund.datas.timeframe import Timeframe


def get_supported_resolutions(product: BaseProduct) -> dict[Timeframe, list[int]]:
    import importlib
    from pfund.enums import Broker

    supported_resolutions: dict[Timeframe, list[int]]
    if product.bkr == Broker.CRYPTO:
        Exchange = getattr(importlib.import_module(f'pfund.brokers.crypto.exchanges.{product.exch.lower()}.exchange'), 'Exchange')
        supported_resolutions = Exchange.get_supported_resolutions(product)
    elif product.bkr == Broker.IBKR:
        InteractiveBrokersAPI = getattr(importlib.import_module('pfund.brokers.ibkr.api'), 'InteractiveBrokersAPI')
        supported_resolutions = InteractiveBrokersAPI.SUPPORTED_RESOLUTIONS
    else:
        raise NotImplementedError(f'broker {product.bkr} is not supported')
    return supported_resolutions