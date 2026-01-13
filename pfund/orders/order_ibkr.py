from typing import Any

from ibapi.order import Order
from ibapi.contract import Contract

from pfund.orders.order_base import BaseOrder


class IBOrder(Order, BaseOrder):
    def model_post_init(self, __context: Any):
        Order.__init__(self)
        # TODO: set IB's order fields
    
    def to_contract(self) -> Contract:
        return self.product.to_contract()