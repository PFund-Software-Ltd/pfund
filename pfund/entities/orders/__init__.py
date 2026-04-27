from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund.entities.orders.order_base import BaseOrder

from pfund.enums import OrderType, TradingVenue


def OrderFactory(venue: TradingVenue | str, order_type: OrderType | str) -> type[BaseOrder]:
    import importlib
    venue = TradingVenue[venue.upper()]
    Order = venue.order_class
    order_type = OrderType[order_type.upper()]
    OrderTypeMixin = getattr(importlib.import_module(f'pfund.entities.orders.mixins.{order_type.lower()}_order'), f'{order_type.capitalize()}OrderMixin')
    class_name = (
        f'{Order.__name__.replace("Order", "")}'
        + OrderTypeMixin.__name__.replace('Mixin', '')
    )
    return type(class_name, (Order, OrderTypeMixin), {"__module__": __name__})