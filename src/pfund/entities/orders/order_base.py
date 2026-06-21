from __future__ import annotations
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

if TYPE_CHECKING:
    from pfund.entities import BaseProduct, BaseAccount
    from pfund.enums import TradingVenue

import math
import time
import logging
from uuid import uuid4
from decimal import Decimal, ROUND_HALF_UP

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

from pfund.entities import Trade, Quantity
from pfund.utils import trim_trailing_zeros
from pfund.enums.order_status import OrderStatus
from pfund.enums import (
    Side,
    OrderType,
    TimeInForce,
)


StrategyName: TypeAlias = str
logger = logging.getLogger("pfund.order_manager")


class BaseOrder(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    creator: StrategyName | Literal["USER"] = Field(
        description="""
        creator of the order, in most cases it should be a strategy name, e.g. xxx_strategy.
        If it is a manual order placed by the user, it is marked as "USER".
        """,
    )
    account: BaseAccount
    product: BaseProduct
    side: Side | Literal["BUY", "SELL", 1, -1]
    quantity: Quantity = Field(gt=0.0)
    price: Decimal | None = Field(default=None, gt=0.0)
    amend_price: Decimal | None = Field(
        default=None, gt=0.0, description="price to amend the order to"
    )
    amend_quantity: Quantity | None = Field(
        default=None, gt=0.0, description="quantity to amend the order to"
    )
    trigger_price: Decimal | None = Field(
        default=None,
        description="""
        price level that activates a conditional order (stop/take-profit);
        when touched, the order converts into a market or limit order.
        """,
    )
    trades: list[Trade] = Field(default_factory=list)
    target_price: Decimal | None = Field(
        default=None,
        description="""
        FOR ANALYSIS ONLY: the price the creator ideally wanted to trade at.
        It is never sent to the trading venue and has no effect on the
        actual order being placed; it only records intent for post-trade analysis.
        e.g. the gap between the fill (avg_price) and target_price measures fill quality
        """,
    )
    order_type: OrderType | str = OrderType.LIMIT
    time_in_force: TimeInForce | str = TimeInForce.GTC
    key: str = Field(
        default="",
        description="""
        your own identifier for the order, set by you when placing it (auto-generated if omitted).
        This is the client order id sent to the trading venue, as opposed to order_id which is
        assigned by the venue.
        """,
    )
    order_id: str = Field(default="", description="order id given by the trading venue")
    reduce_only: bool = False
    remark: str = ""

    def model_post_init(self, __context: Any):
        self.key = self.key or self._generate_key()
        self.quantity = self._round_to_lot(self.quantity)
        if self.price is not None:
            self.price = self._round_to_tick(self.price, rounding="passive")
        if self.trigger_price is not None:
            self.trigger_price = self._round_to_tick(
                self.trigger_price, rounding="nearest"
            )
        # TODO
        # self._status = [None] * 4
        # self._status_reasons = {}  # { MainOrderStatus: reason}
        # self.timestamps = {}  # { MainOrderStatus.SUBMITTED: ts }

    @field_validator("side", mode="before")
    @classmethod
    def _validate_side(cls, value: Any) -> Side:
        if isinstance(value, str):
            side = Side[value.upper()]
        elif isinstance(value, int):
            side = Side(value)
        else:
            raise ValueError(f"Invalid side: {value}")
        return side

    @field_validator("order_type", mode="before")
    @classmethod
    def _validate_order_type(cls, value: Any) -> OrderType:
        if isinstance(value, str):
            return OrderType[value.upper()]
        raise ValueError(f"Invalid order_type: {value}")

    @field_validator("time_in_force", mode="before")
    @classmethod
    def _validate_time_in_force(cls, value: Any) -> TimeInForce:
        if isinstance(value, str):
            return TimeInForce[value.upper()]
        raise ValueError(f"Invalid time_in_force: {value}")

    @staticmethod
    def _generate_key() -> str:
        return str(uuid4())

    def _round_to_lot(self, quantity: Quantity) -> Quantity:
        if self.lot_size is None:
            return quantity
        lot_size = self.lot_size
        rounded_quantity = math.floor(quantity / lot_size) * lot_size
        return Quantity(trim_trailing_zeros(rounded_quantity))

    def _round_to_tick(
        self,
        price: Decimal,
        rounding: Literal["nearest", "passive", "aggressive"] = "nearest",
    ) -> Decimal:
        if self.tick_size is None:
            return price
        tick_size = self.tick_size
        steps = price / tick_size
        if rounding == "nearest":
            adj_steps = steps.to_integral_value(rounding=ROUND_HALF_UP)
        elif rounding == "passive":  # favorable price (buy down / sell up)
            adj_steps = math.floor(steps) if self.side == 1 else math.ceil(steps)
        elif rounding == "aggressive":  # unfavorable price (buy up / sell down)
            adj_steps = math.ceil(steps) if self.side == 1 else math.floor(steps)
        else:
            raise ValueError(f"Invalid rounding mode: {rounding}")
        return trim_trailing_zeros(adj_steps * tick_size)

    @computed_field
    @property
    def size(self) -> Quantity:
        return self.quantity * Side(self.side)

    @property
    def last_traded_price(self) -> Decimal | None:
        if not self.is_traded():
            return None
        return self.trades[-1].price

    ltp = last_traded_price

    @property
    def last_traded_size(self) -> Quantity:
        return self.last_traded_quantity * Side(self.side)

    lts = last_traded_size

    @property
    def last_traded_quantity(self) -> Quantity:
        if not self.is_traded():
            return Quantity(0)
        return self.trades[-1].quantity

    last_traded_qty = ltq = last_traded_quantity

    @property
    def qty(self) -> Quantity:
        return self.quantity

    @property
    def amend_qty(self) -> Quantity | None:
        return self.amend_quantity

    @property
    def px(self) -> Decimal | None:
        return self.price

    @property
    def amend_px(self) -> Decimal | None:
        return self.amend_price

    @property
    def trigger_px(self) -> Decimal | None:
        return self.trigger_price

    @property
    def target_px(self) -> Decimal | None:
        return self.target_price

    @computed_field
    @property
    def filled_size(self) -> Quantity:
        return self.filled_quantity * Side(self.side)

    @property
    def filled_quantity(self) -> Quantity:
        return sum((trade.quantity for trade in self.trades), Quantity(0))

    filled_qty = filled_quantity

    @property
    def avg_price(self) -> Decimal | None:
        filled_quantity = self.filled_quantity
        if filled_quantity == 0:
            return None
        notional = sum(
            (trade.price * trade.quantity for trade in self.trades), Decimal(0)
        )
        return notional / filled_quantity

    avg_px = avg_price

    @property
    def remaining_size(self) -> Quantity:
        return self.remaining_quantity * Side(self.side)

    @property
    def remaining_quantity(self) -> Quantity:
        return self.quantity - self.filled_quantity

    remaining_qty = remaining_quantity

    @property
    def venue(self) -> TradingVenue:
        assert self.product.venue is not None, "product venue is not set"
        return self.product.venue

    @property
    def tick_size(self) -> Decimal | None:
        return self.product.tick_size

    @property
    def lot_size(self) -> Decimal | None:
        return self.product.lot_size

    @property
    def tif(self):
        return self.time_in_force

    @property
    def type(self):
        return self.order_type

    @property
    def id(self):
        return self.order_id

    def is_traded(self) -> bool:
        return bool(self.trades)

    def is_filled(self):
        return self.filled_qty == self.qty

    def is_amending(self):
        return self._status[OrderStatus.Amend.position] == OrderStatus.Amend.SUBMITTED

    def is_cancelling(self):
        return self._status[OrderStatus.Cancel.position] == OrderStatus.Cancel.SUBMITTED

    def is_opened(self):
        return self._status[OrderStatus.Main.position] == OrderStatus.Main.OPENED

    def is_closed(self):
        return not self.is_opened()

    def amend(self, *, price: Decimal | None = None, quantity: Quantity | None = None):
        if price is None and quantity is None:
            raise ValueError("amend() requires price and/or quantity")
        if quantity is not None:
            self.amend_quantity = self._round_to_lot(quantity)
        if price is not None:
            self.amend_price = self._round_to_tick(price, rounding="passive")
        # TODO: self.on_status_update(OrderStatus.Amend.SUBMITTED) once the status machine is wired

    def add_trade(self, trade: Trade):
        self.trades.append(trade)

    def on_status_update(self, status, ts=None, reason="") -> bool:
        is_updated = False
        prev_status = self._status
        status_type = type(status)
        self._status[status_type.position] = status
        self._status_reasons[status_type] = reason
        self.timestamps[status] = ts or time.time()
        if prev_status != self._status:
            is_updated = True
            logger.debug(repr(self))
        return is_updated

    # TODO:
    # def on_trade_update(
    #     self, avg_price, filled_qty, last_traded_px, last_traded_qty
    # ) -> bool:
    #     prev_avg_price = self.avg_price if self.avg_price else 0.0
    #     prev_filled_qty = self.filled_qty
    #     is_updated = False
    #     if not filled_qty:
    #         logger.error(
    #             f"trade update has {filled_qty=} {self.creator=} {self.bkr} {self.exch} {self.oid=}"
    #         )
    #     else:
    #         if filled_qty > prev_filled_qty:
    #             is_updated = True
    #             self.filled_qty = filled_qty
    #             self.last_traded_qty = self.ltq = filled_qty - prev_filled_qty
    #             self.remain_qty = self.quantity - filled_qty
    #             # prev_avg_price * prev_filled_qty + last_traded_qty * last_traded_px = avg_price * filled_qty
    #             if avg_price:
    #                 self.avg_price = avg_price
    #                 # NOTE: derived last_traded_px
    #                 self.last_traded_px = self.ltp = (
    #                     avg_price * self.filled_qty - prev_avg_price * prev_filled_qty
    #                 ) / self.last_traded_qty
    #             elif last_traded_px:
    #                 self.last_traded_px = self.ltp = last_traded_px
    #                 # NOTE: derived avg_price
    #                 self.avg_price = (
    #                     prev_avg_price * prev_filled_qty + self.ltq * last_traded_px
    #                 ) / self.filled_qty
    #             else:
    #                 # NOTE: assumed avg_price and last_traded_px to be order price
    #                 self.avg_price = self.ltp = self.price
    #             self.trades.append({"px": self.ltp, "qty": self.ltq})
    #         elif filled_qty < prev_filled_qty:
    #             logger.warning(
    #                 f"Delayed trade msg {self.creator=} {self.bkr} {self.exch} {self.oid=} ({filled_qty=} < {prev_filled_qty=})"
    #             )
    #     return is_updated

    def get_status(self, mode: Literal["abbrev", "detailed", "standard"] = "standard"):
        """Returns order status in differet modes.
        Args:
            mode:
                1. if mode='abbrev', returns sth like 'O---', which is an abbreviation of 4 types of order statuses
                2. if mode='detailed', it converts the abbreviation into human-readable string, e.g. 'OPENED,PARTIAL,SUBMITTED,AMENDED'
                3. if mode='standard', returns only the crucial info: CREATED/OPENED/PARTIAL/FILLED/CLOSED/CANCELLED
        """
        if mode == "abbrev":
            return "".join(
                [status.name[0] if status else "-" for status in self._status]
            )
        elif mode == "detailed":
            readable_o_status = []
            for status in self._status:
                if status is None:
                    continue
                status_str = type(status).__name__
                readable_o_status.append(status_str + " " + status.name)
            if readable_o_status:
                return " | ".join(readable_o_status)
        elif mode == "standard":
            for status in self._status:
                if (status is not None and type(status) is not OrderStatus.Cancel) or (
                    status == OrderStatus.Cancel.CANCELLED
                ):
                    o_status = status.name
            return o_status

    def print_status(self):
        readable_o_status = self.get_status(mode="detailed")
        print(f"Order Status(id={self.id}): {readable_o_status}")

    def __str__(self):
        side_str = "BUY" if self.side == 1 else "SELL"
        return (
            f"Strategy={self.creator}|Broker={self.bkr}|Exchange={self.exch}|Account={self.acc}|Product={self.pdt}\n"
            f"OrderType={self.type}|TimeInForce={self.tif}|IsReduceOnly={self.reduce_only}\n"
            f"Side={side_str}|Price={self.price}|Quantity={self.quantity}\n"
            f"AveragePrice={self.avg_price}|FilledQuantity={self.filled_qty}\n"
            f"TriggerPrice={self.trigger_px}|TargetPrice={self.target_px}\n"
            f"AmendPrice={self.amend_px}|AmendQuantity={self.amend_qty}"
        )

    def __repr__(self):
        filled_size = self.filled_qty * self.side
        last_traded_size = self.ltq * self.side
        amend_size = self.amend_qty * self.side
        status_abbrev = self.get_status()
        return (
            f"{self.creator}|{self.venue}|{self.acc}|{self.pdt}|{self.oid}|{self.eoid}|"
            f"{self.type}|{self.tif}|{status_abbrev}|{self.size}@{self.price}|"
            f"filled={filled_size}@{self.avg_price}|"
            f"last={last_traded_size}@{self.ltp}|"
            f"amend={amend_size}@{self.amend_px}|"
            f"trigger={self.trigger_px}|target={self.target_px}|"
            f"reduce_only={self.reduce_only}"
        )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, BaseOrder):
            return NotImplemented  # Allow other types to define equality with BaseOrder
        return self.key == other.key

    def __hash__(self) -> int:
        return hash(self.key)
