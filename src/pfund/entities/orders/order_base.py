from __future__ import annotations
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, TypedDict, ClassVar

if TYPE_CHECKING:
    from pfund.venues.adapter_base import BaseAdapter
    from pfund.entities import BaseProduct, BaseAccount
    from pfund.enums import TradingVenue

    # TODO
    class OrderUpdate(TypedDict):
        source: Literal[
            "response",  # e.g. place_orders/cancel_orders RESTful API response
            "order_event",  # e.g. order update from websocket
            "trade_event",  # e.g. trade update from websocket
            "get_trade_history",  # update from a specific reconciliation method
            "get_active_orders",  # update from a specific reconciliation method
        ]


import math
from uuid import uuid4
from decimal import Decimal, ROUND_HALF_UP
import builtins

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)

from pfund.entities.orders.trailing_stop import TrailingStop
from pfund.entities.trades.trade import Trade
from pfund.entities.trades.quantity import Quantity
from pfund.utils import trim_trailing_zeros
from pfund.entities.orders.order_status import OrderStatus
from pfund.enums import (
    Side,
    OrderType,
    TimeInForce,
)


OrderKey: TypeAlias = str  # order key = client order id
StrategyName: TypeAlias = str


class BaseOrder(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    # NOTE: use builtins.type to avoid collision with property "type"
    TrailingStop: ClassVar[builtins.type[TrailingStop]] = TrailingStop
    # Some venues (most crypto REST/WS APIs) never send a separate "pending"
    # acknowledgement — they jump straight from our local SUBMITTED to a terminal
    # report. Set True on those venue subclasses to treat SUBMITTED as PENDING, so
    # is_cancelling()/is_amending() and the derived FIX status reflect the in-flight
    # request instead of waiting on an ack that will never arrive.
    SUBMITTED_AS_PENDING: ClassVar[bool] = False

    creator: StrategyName | Literal["USER"] = Field(
        description="""
        creator of the order, in most cases it should be a strategy name, e.g. xxx_strategy.
        If it is a manual order placed by the user, it is marked as "USER".
        """,
    )
    account: BaseAccount
    product: BaseProduct
    status: OrderStatus = Field(default_factory=OrderStatus)
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
        For a trailing stop this is the CURRENT stop level, recomputed as the
        market moves (by the venue natively, or by the simulator in backtest).
        """,
    )
    trigger_direction: Literal["up", "down"] | None = Field(
        default=None,
        description="""
        REQUIRED for STOP_MARKET/STOP_LIMIT (a trailing stop is exempt — it derives
        direction from side). Which way the market must move to fire the trigger:
          'up'   = fire when price RISES to trigger_price
          'down' = fire when price FALLS to trigger_price
        """,
    )
    trailing_stop: TrailingStop | None = Field(
        default=None,
        description="""
        the trailing rule (offset only); when set, trigger_price is produced
        by this rule rather than supplied directly. Direction comes from side.
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
    order_type: OrderType | str = Field(
        default=OrderType.LIMIT,
        description="""
        LIMIT/MARKET: plain resting-limit / immediate-market orders.
        STOP_MARKET/STOP_LIMIT: conditional orders that stay dormant until
        trigger_price is touched (in the trigger_direction), then fire as a
        MARKET / LIMIT order respectively.
        """,
    )
    time_in_force: TimeInForce | str = TimeInForce.GTC
    key: OrderKey = Field(
        default="",
        description="""
        your own identifier for the order, set by you when placing it (auto-generated if omitted).
        This is the client order id sent to the trading venue, as opposed to order_id which is
        assigned by the venue.
        """,
    )
    order_id: str = Field(default="", description="order id given by the trading venue")
    post_only: bool = Field(
        default=False,
        description="Only add liquidity as a maker; reject/cancel if the order would immediately match (take liquidity).",
    )
    reduce_only: bool = Field(
        default=False,
        description="Only reduce an existing position; never open, increase, or flip it. Excess size is trimmed or cancelled.",
    )
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
        if self.trailing_stop is not None:
            if self.trailing_stop.amount is not None:
                self.trailing_stop.amount = self._round_to_tick(
                    self.trailing_stop.amount, rounding="nearest"
                )
            if self.trailing_stop.limit_offset is not None:
                self.trailing_stop.limit_offset = self._round_to_tick(
                    self.trailing_stop.limit_offset, rounding="nearest"
                )

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
    def _validate_order_type(cls, value: Any) -> OrderType | str:
        if isinstance(value, str):
            return OrderType.__members__.get(value.upper(), value)
        return value

    @field_validator("time_in_force", mode="before")
    @classmethod
    def _validate_time_in_force(cls, value: Any) -> TimeInForce | str:
        if isinstance(value, str):
            return TimeInForce.__members__.get(value.upper(), value)
        return value

    @model_validator(mode="after")
    def _validate_trigger_price(self) -> BaseOrder:
        # a trailing stop produces trigger_price and derives direction from side,
        # so it's exempt here (handled by _validate_trailing_stop instead)
        if self.is_stop() and self.trailing_stop is None:
            if self.trigger_price is None:
                raise ValueError(
                    f"trigger_price is required for order_type {self.order_type}"
                )
            if self.trigger_direction is None:
                raise ValueError(
                    "trigger_direction ('up'/'down') is required for order_type "
                    + f"{self.order_type}; pfund never infers it from a reference price"
                )
        return self

    @model_validator(mode="after")
    def _validate_trailing_stop(self) -> BaseOrder:
        if self.trailing_stop is None:
            return self

        # TODO: trailing stop is not supported yet — it requires implementing the
        # trailing mechanics in BOTH the simulated venue (backtest, sandbox env) and live trading.
        # Until then, fail loudly rather than silently send an unsupported order.
        raise NotImplementedError("trailing_stop is not supported yet")

        if self.order_type not in (OrderType.STOP_MARKET, OrderType.STOP_LIMIT):
            raise ValueError(
                "trailing_stop requires order_type STOP_MARKET or STOP_LIMIT; "
                + f"got {self.order_type}"
            )
        if self.trigger_price is not None:
            raise ValueError(
                "trigger_price cannot be provided with trailing_stop; "
                + "it is produced by the trailing rule as the market moves"
            )
        if self.price is not None:
            raise ValueError(
                "price cannot be provided with trailing_stop; "
                + "the limit price is produced on activation (trigger -/+ limit_offset)"
            )
        if (
            self.order_type == OrderType.STOP_LIMIT
            and self.trailing_stop.limit_offset is None
        ):
            raise ValueError(
                "trailing_stop with order_type STOP_LIMIT requires limit_offset; "
                + "it sets the limit price (trigger -/+ limit_offset) on activation"
            )
        return self

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

    @property
    def adapter(self) -> BaseAdapter:
        return self.venue.venue_class.adapter

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
    def amend_size(self) -> Quantity | None:
        if self.amend_quantity is None:
            return None
        return self.amend_quantity * Side(self.side)

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
    def trigger_dir(self) -> Literal["up", "down"] | None:
        return self.trigger_direction

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
        return self.product.market.tick_size if self.product.market else None

    @property
    def lot_size(self) -> Decimal | None:
        return self.product.market.lot_size if self.product.market else None

    @property
    def tif(self):
        return self.time_in_force

    @property
    def type(self):
        return self.order_type

    @property
    def id(self):
        return self.order_id

    def is_stop(self) -> bool:
        return self.order_type in (OrderType.STOP_LIMIT, OrderType.STOP_MARKET)

    def is_traded(self) -> bool:
        return bool(self.trades)

    def is_partially_filled(self) -> bool:
        return self.filled_qty < self.qty

    is_partial = is_partially_filled

    def is_filled(self):
        return self.filled_qty == self.qty

    def is_submitted(self):
        return self.status.is_submitted()

    def is_pending(self):
        return self.status.is_pending()

    def is_active(self):
        return self.status.is_active()

    def is_closed(self):
        return self.status.is_closed()

    def is_cancelling(self):
        return self.status.is_cancelling(include_submitted=self.SUBMITTED_AS_PENDING)

    def is_cancelled(self):
        return self.status.is_cancelled()

    def is_amending(self):
        return self.status.is_amending(include_submitted=self.SUBMITTED_AS_PENDING)

    def is_amended(self):
        return self.status.is_amended()

    def amend(self, *, price: Decimal | None = None, quantity: Quantity | None = None):
        if price is None and quantity is None:
            raise ValueError("amend() requires price and/or quantity")
        if quantity is not None:
            self.amend_quantity = self._round_to_lot(quantity)
        if price is not None:
            self.amend_price = self._round_to_tick(price, rounding="passive")

    def add_trade(self, trade: Trade):
        self.trades.append(trade)

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

    def __str__(self):
        return (
            f"Creator={self.creator} | Venue={self.venue} | Account={self.account.name} | Product={self.product.name}\n"
            f"OrderKey={self.key} | OrderID={self.id} | FIXStatus={self.status.FIX} | OrderStatus={self.status}\n"
            f"OrderType={self.type} | TimeInForce={self.tif} | ReduceOnly={self.reduce_only}\n"
            f"Side={Side(self.side).name} | Price={self.price} | Quantity={self.quantity}\n"
            f"AveragePrice={self.avg_price} | FilledQuantity={self.filled_qty}\n"
            f"TriggerPrice={self.trigger_px} | TargetPrice={self.target_px}\n"
            f"AmendPrice={self.amend_px} | AmendQuantity={self.amend_qty}"
        )

    def __repr__(self):
        return (
            f"{self.creator}|{self.venue}|{self.account.name}|{self.product.name}|"
            f"{self.key}|{self.id}|{self.status.FIX}|{repr(self.status)}|"
            f"{self.type}|{self.tif}|{self.size}@{self.price}|"
            f"filled={self.filled_size}@{self.avg_price}|"
            f"last={self.lts}@{self.ltp}|"
            f"amend={self.amend_size}@{self.amend_price}|"
            f"trigger={self.trigger_price}|target={self.target_price}|"
            f"reduce_only={self.reduce_only}|remark={self.remark}"
        )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, BaseOrder):
            return NotImplemented  # Allow other types to define equality with BaseOrder
        return self.key == other.key

    def __hash__(self) -> int:
        return hash(self.key)
