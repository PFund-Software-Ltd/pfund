from typing import Any

from ibapi.contract import Contract
from ibapi.order import Order as IBOrder

from pfund.entities.orders.order_base import BaseOrder


class InteractiveBrokersOrder(BaseOrder, IBOrder):
    def model_post_init(self, __context: Any):
        IBOrder.__init__(self)
        super().model_post_init(__context)
        # TODO: set IB's order fields

    def to_contract(self) -> Contract:
        pass
