from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pfund.entities import BaseOrder
    from pfund.entities.orders.order_base import OrderUpdate, OrderKey


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
    # TODO: manage trades (BaseTrade) as well
    def __init__(self):
        self._logger = logging.getLogger("pfund.order_manager")
        self._submitted_orders: dict[OrderKey, BaseOrder] = {}
        self._active_orders: dict[OrderKey, BaseOrder] = {}
        self._closed_orders: dict[OrderKey, BaseOrder] = {}
        self._counters: dict[OrderKey, OrderCounter] = defaultdict(OrderCounter)

    # TODO: get orders by venue, by status etc.
    def get_orders(self):
        pass

    def update_orders(self, update: OrderUpdate):
        pass

    # TODO: also reconcile with strategies
    # def reconcile_orders(self):
    #     def work():
    #         for exch in self._accounts:
    #             for acc in self._accounts[exch]:
    #                 self.get_orders(exch, acc, is_api_call=True)

    #     func = inspect.stack()[0][3]
    #     Thread(target=work, name=func + "_thread", daemon=True).start()

    # TODO: also reconcile with strategies
    # def reconcile_trades(self):
    #     def work():
    #         for exch in self._accounts:
    #             for acc in self._accounts[exch]:
    #                 self.get_trades(exch, acc, is_api_call=True)

    #     func = inspect.stack()[0][3]
    #     Thread(target=work, name=func + "_thread", daemon=True).start()
