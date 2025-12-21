from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund.typing import tOrderType, tTradingVenue
    from pfund.orders.order_base import BaseOrder

from pfund.enums import OrderType, TradingVenue


def OrderFactory(trading_venue: TradingVenue | tTradingVenue, order_type: OrderType | tOrderType) -> type[BaseOrder]:
    import importlib
    trading_venue = TradingVenue[trading_venue.upper()]
    Order = trading_venue.order_class
    order_type = OrderType[order_type.upper()]
    OrderTypeMixin = getattr(importlib.import_module(f'pfund.orders.mixins.{order_type.lower()}_order'), f'{order_type.capitalize()}OrderMixin')
    class_name = (
        f'{Order.__name__.replace("Order", "")}'
        + OrderTypeMixin.__name__.replace('Mixin', '')
    )
    return type(class_name, (Order, OrderTypeMixin), {"__module__": __name__})