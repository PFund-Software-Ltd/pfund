from typing import Any, ClassVar

import logging

from ibapi.order import Order as IBOrder

from pfund.entities.orders.order_base import BaseOrder
from pfund.enums import OrderType, Side


logger = logging.getLogger("pfund.ibkr")


class InteractiveBrokersOrder(BaseOrder, IBOrder):  # pyright: ignore[reportUnsafeMultipleInheritance]
    # NOTE: only used as reference, not used at runtime
    COMMON_ORDER_TYPES: ClassVar[dict[str, str]] = {
        "MIT": "Market-If-Touched",
        "MOO": "Market-On-Open",
        "MOC": "Market-On-Close",
        "LIT": "Limit-If-Touched",
        "LOO": "Limit-On-Open",
        "LOC": "Limit-On-Close",
        "TRAIL": "Trailing-Stop-Market",
        "TRAIL LIMIT": "Trailing-Stop-Limit",
    }

    def model_post_init(self, __context: Any):
        IBOrder.__init__(self)
        super().model_post_init(__context)
        # TODO: set IB's order fields
        if self.is_stop():
            self.order_type = self._resolve_stop_order_type()

    def _resolve_stop_order_type(self) -> OrderType | str:
        """Resolve the IBKR native order type for a stop order.

        IBKR stop market/limit orders are for STOP-LOSS only. If someone assumes a
        stop order in IBKR works like it does on e.g. binance/bybit, a take-profit
        (user's intention) stop order will get triggered right away.

        Logic Walkthrough (SELL take-profit on a long, e.g. market price=100, want to sell at 110):
        - take-profit intention puts the trigger on the FAVORABLE side, i.e. ABOVE the
          market for a SELL (110 > 100).
        - but IBKR fixes a STP order's trigger direction by side: a SELL STP is a
          stop-loss, so it fires when the market FALLS to/through the stop, i.e.
          condition is "last price <= stop price".
        - our trigger (110) is already above the market (100), so "100 <= 110" is
          ALREADY TRUE the moment IBKR receives it -> it triggers instantly and fires
          as a market order at ~100, not 110. (a BUY take-profit below market breaks
          the same way: BUY STP fires when "last >= stop", already true.)
        - fix: the favorable side must use MIT/LIT instead, whose trigger direction is
          the OPPOSITE (a SELL MIT fires when the market RISES to the touch price), so
          it correctly waits until 110 is reached.
        """
        if not self.is_stop():
            return self.order_type
        # NOTE: stop-loss/take-profit is just a convention,
        # it could be just a conditional order without any existing position
        # to be precise, stop-loss = adverse condition, take-profit = favorable condition
        is_stop_loss = (self.trigger_direction == "up" and self.side == Side.BUY) or (
            self.trigger_direction == "down" and self.side == Side.SELL
        )
        is_take_profit = (
            self.trigger_direction == "up" and self.side == Side.SELL
        ) or (self.trigger_direction == "down" and self.side == Side.BUY)
        if is_stop_loss:
            order_type = self.adapter(self.type, group="order_types")
        elif is_take_profit:
            is_limit = self.order_type == OrderType.STOP_LIMIT
            order_type = "LIT" if is_limit else "MIT"
            logger.warning(
                f"{self.order_type} on the favorable side (take-profit) is automatically mapped to '{order_type}'; "
                + "Without this guard, a native IBKR stop ('STP'/'STP LMT') would trigger instantly"
            )
        else:
            order_type = self.order_type
        return order_type
