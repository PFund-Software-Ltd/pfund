from __future__ import annotations
from typing import ClassVar, TYPE_CHECKING, TypeAlias, TypedDict, Literal

if TYPE_CHECKING:
    from pfund.entities import BaseOrder

    OrderKey: TypeAlias = str  # order key = client order id

    # TODO
    class OrderUpdate(TypedDict):
        source: Literal[
            "response",  # e.g. place_orders/cancel_orders RESTful API response
            "order_event",  # e.g. order update from websocket
            "trade_event",  # e.g. trade update from websocket
            "get_trade_history",  # update from a specific reconciliation method
            "get_active_orders",  # update from a specific reconciliation method
        ]


import logging
from dataclasses import dataclass
from collections import defaultdict

import apscheduler

from pfund.entities.orders.order_status import OrderStatus


@dataclass
class OrderCounter:
    missed: int = 0
    reconciled: int = 0
    cancel_rejected: int = 0


class OrderManager:
    def __init__(self):
        self._logger = logging.getLogger("pfund.order_manager")
        self.submitted_orders: dict[OrderKey, BaseOrder] = {}
        self.active_orders: dict[OrderKey, BaseOrder] = {}
        self.closed_orders: dict[OrderKey, BaseOrder] = {}
        self._counters: dict[OrderKey, OrderCounter] = defaultdict(OrderCounter)
