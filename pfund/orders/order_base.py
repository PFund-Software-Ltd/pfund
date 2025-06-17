from __future__ import annotations
from typing import TYPE_CHECKING, Literal, Any
if TYPE_CHECKING:
    from pfund.enums import TradingVenue, Broker, CryptoExchange
    from pfund.products.product_base import BaseProduct
    from pfund.accounts.account_base import BaseAccount

import time
import logging
import hashlib
import math
from uuid import uuid4
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, model_validator, field_validator

from pfund.enums import (
    MainOrderStatus, 
    FillOrderStatus, 
    CancelOrderStatus, 
    AmendOrderStatus, 
    OrderSide, 
    OrderType,
    TimeInForce
)

# Indices in order status, e.g. order status = 'S---', index 0 represents the MainOrderStatus etc.
STATUS_INDEX = {
    MainOrderStatus: 0,
    0: MainOrderStatus,
    FillOrderStatus: 1,
    1: FillOrderStatus,
    CancelOrderStatus: 2,
    2: CancelOrderStatus,
    AmendOrderStatus: 3,
    3: AmendOrderStatus
}

ORDER_SIDE = {
    OrderSide.BUY: 1,
    1: OrderSide.BUY,
    OrderSide.SELL: -1,
    -1: OrderSide.SELL,
}
logger = logging.getLogger('orders')


class BaseOrder(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid')

    creator: str = Field(default='', description="creator of the order, e.g. strategy's name, agent's name, etc.")
    account: BaseAccount
    product: BaseProduct
    side: OrderSide | None = None
    size: Decimal | None = Field(default=None, description="size = signed quantity = quantity * side")
    quantity: Decimal = Field(default=0.0, gt=0.0)
    price: Decimal | None = Field(default=None, gt=0.0)
    trigger_price: Decimal | None = None
    target_price: Decimal | None = None   # used for slippage calculation
    order_type: OrderType | str = OrderType.LIMIT
    time_in_force: TimeInForce | str = TimeInForce.GTC
    is_reduce_only: bool = False
    key: str | None=Field(default=None, description='unique key for the order')
    order_id: str=Field(default='', description='order id given by the trading venue')
    alias: str=''
    remark: str = ''
    
    def model_post_init(self, __context: Any):
        self.size = self.quantity * self.side
        self.key = self.key or self._generate_key()
        # TODO
        # self.quantity = self._adjust_by_lot_size(qty)
        # self.prev_qty = self.remain_qty = self.quantity
        # self.filled_qty = self.last_traded_qty = self.ltq = 0.0
        # self.amend_qty = None
        # self.price = self._adjust_by_tick_size(px) if px else None
        # self.prev_px = self.price
        # self.trigger_px = self._adjust_by_tick_size(trigger_px) if trigger_px else None
        # self.target_px = self._adjust_by_tick_size(target_px) if target_px else None
        # self.avg_px = self.last_traded_px = self.ltp = None
        # self.amend_px = None
        # self._status = [None] * 4
        # self._status_reasons = {}  # { MainOrderStatus: reason}
        # self.timestamps = {}  # { MainOrderStatus.SUBMITTED: ts }
        # self.trades = []

    def _generate_key(self):
        # return hashlib.md5((self.creator + str(time.time())).encode()).hexdigest()
        return uuid4()
    
    @property
    def tv(self) -> TradingVenue:
        return self.trading_venue
    
    @property
    def trading_venue(self) -> TradingVenue:
        return self.product.trading_venue
    
    @property
    def bkr(self) -> Broker:
        return self.broker
    
    @property
    def broker(self) -> Broker:
        return self.product.broker
    
    @property
    def exch(self) -> CryptoExchange | str:
        return self.exchange
    
    @property
    def exchange(self) -> CryptoExchange | str:
        return self.product.exchange

    @property
    def adapter(self):
        return self.product.adapter
    
    @property
    def tick_size(self):
        return self.product.tick_size
    
    @property
    def lot_size(self):
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
    
    @property
    def is_inverse(self):
        return self.product.is_inverse()

    def _create_side(self, side: OrderSide | Literal['BUY', 'SELL'] | Literal[1, -1]) -> OrderSide:
        if isinstance(side, int):
            side = ORDER_SIDE[side]
        return OrderSide[side.upper()]
    
    @property
    def linear_size(self):
        if px := self.price if self.is_inverse() else 1:
            return self.size * self.product.multi / px
    
    @property
    def linear_qty(self):
        return abs(self.linear_size)
    
    @property
    def linear_filled_qty(self):
        if avg_px := self.avg_px if self.is_inverse() else 1:
            return self.filled_qty * self.product.multi / avg_px

    @property
    def linear_remain_qty(self):
        return self.linear_qty - self.linear_filled_qty

    # NOTE: Q is shorthand for expressing values in the quote asset as the unit
    @property
    def qtyQ(self):
        if not self.is_inverse():
            if self.filled_qtyQ is None or self.remain_qtyQ is None:
                return None
            else:
                return self.filled_qtyQ + self.remain_qtyQ
    
    @property
    def sizeQ(self):
        if not self.is_inverse():
            return self.quantityQ * -self.side

    @property
    def filled_qtyQ(self):
        if not self.is_inverse():
            if not self.avg_px:
                return Decimal(0) if self.filled_qty == 0 else None
            else:
                return self.avg_px * self.filled_qty
    
    @property
    def remain_qtyQ(self):
        if not self.is_inverse():
            if not self.price:
                return Decimal(0) if self.remain_qty == 0 else None
            else:
                return self.price * self.remain_qty
    
    def _adjust_by_tick_size(self, px: float|str):
        math_func = math.floor if self.side == 1 else math.ceil
        tick_size = self.product.tsize
        px = Decimal(str(px))
        adj_px = Decimal(str(math_func(px / tick_size) * tick_size))
        # trim unnecessary zeros e.g. 25000.00 -> 25000
        adj_px = adj_px.to_integral() if adj_px == adj_px.to_integral() else adj_px.normalize()
        return adj_px
    
    def _adjust_by_lot_size(self, qty: float|str):
        lot_size = self.product.lsize
        qty = Decimal(str(qty))
        adj_qty = Decimal(str(math.floor(qty / lot_size) * lot_size))
        # trim unnecessary zeros e.g. 0.010 -> 0.01
        adj_qty = adj_qty.to_integral() if adj_qty == adj_qty.to_integral() else adj_qty.normalize()
        return adj_qty
    
    def is_amending(self):
        status_index = STATUS_INDEX[AmendOrderStatus]
        return self._status[status_index] == AmendOrderStatus.SUBMITTED

    def is_cancelling(self):
        status_index = STATUS_INDEX[CancelOrderStatus]
        return self._status[status_index] == CancelOrderStatus.SUBMITTED

    def is_opened(self):
        status_index = STATUS_INDEX[MainOrderStatus]
        return self._status[status_index] == MainOrderStatus.OPENED

    def is_closed(self):
        return not self.is_opened()
    
    def is_traded(self):
        return self.filled_qty > 0
    
    def is_filled(self):
        return self.filled_qty == self.quantity

    def on_status_update(self, status, ts=None, reason='') -> bool:
        is_updated = False
        prev_status = self._status
        status_type = type(status)
        status_index = STATUS_INDEX[status_type]
        self._status[status_index] = status
        self._status_reasons[status_type] = reason
        self.timestamps[status] = ts or time.time()
        if prev_status != self._status:
            is_updated = True
            logger.debug(repr(self))
        return is_updated

    def on_trade_update(self, avg_px, filled_qty, last_traded_px, last_traded_qty) -> bool:
        prev_avg_px = self.avg_px if self.avg_px else 0.0
        prev_filled_qty = self.filled_qty
        is_updated = False
        if not filled_qty:
            logger.error(f'trade update has {filled_qty=} {self.creator=} {self.bkr} {self.exch} {self.oid=}')
        else:
            if filled_qty > prev_filled_qty:
                is_updated = True
                self.filled_qty = filled_qty
                self.last_traded_qty = self.ltq = filled_qty - prev_filled_qty
                self.remain_qty = self.quantity - filled_qty
                # prev_avg_px * prev_filled_qty + last_traded_qty * last_traded_px = avg_px * filled_qty
                if avg_px:
                    self.avg_px = avg_px
                    # NOTE: derived last_traded_px
                    self.last_traded_px = self.ltp = (avg_px * self.filled_qty - prev_avg_px * prev_filled_qty) / self.last_traded_qty
                elif last_traded_px:
                    self.last_traded_px = self.ltp = last_traded_px
                    # NOTE: derived avg_px
                    self.avg_px = (prev_avg_px * prev_filled_qty + self.ltq * last_traded_px) / self.filled_qty
                else:
                    # NOTE: assumed avg_px and last_traded_px to be order price
                    self.avg_px = self.ltp = self.price
                self.trades.append({'px': self.ltp, 'qty': self.ltq})
            elif filled_qty < prev_filled_qty:
                logger.warning(f'Delayed trade msg {self.creator=} {self.bkr} {self.exch} {self.oid=} ({filled_qty=} < {prev_filled_qty=})')
        return is_updated
    
    def get_status(self, mode: Literal['abbrev', 'detailed', 'standard']='standard'):
        '''Returns order status in differet modes.
        Args:
            mode: 
                1. if mode='abbrev', returns sth like 'O---', which is an abbreviation of 4 types of order statuses
                2. if mode='detailed', it converts the abbreviation into human-readable string, e.g. 'OPENED,PARTIAL,SUBMITTED,AMENDED'
                3. if mode='standard', returns only the crucial info: CREATED/OPENED/PARTIAL/FILLED/CLOSED/CANCELLED
        '''
        if mode == 'abbrev':
            return ''.join([status.name[0] if status else '-' for status in self._status])
        elif mode == 'detailed':
            readable_o_status = []
            for status in self._status:
                if status is None:
                    continue
                status_str = type(status).__name__.split('OrderStatus')[0]
                readable_o_status.append(status_str + ' ' + status.name)
            if readable_o_status:
                return ' | '.join(readable_o_status)
        elif mode == 'standard':
            for status in self._status:
                if (status is not None and type(status) is not CancelOrderStatus) or \
                    (status == CancelOrderStatus.CANCELLED):
                    o_status = status.name
            return o_status

    def print_status(self):
        readable_o_status = self.get_status(mode='detailed')
        print(f'Order Status(id={self.id}): {readable_o_status}')          

    def __str__(self):
        side_str = 'BUY' if self.side == 1 else 'SELL'
        return f'Strategy={self.creator}|Broker={self.bkr}|Exchange={self.exch}|Account={self.acc}|Product={self.pdt}\n' \
               f'OrderType={self.type}|TimeInForce={self.tif}|IsReduceOnly={self.is_reduce_only}\n' \
               f'Side={side_str}|Price={self.price}|Quantity={self.quantity}\n' \
               f'AveragePrice={self.avg_px}|FilledQuantity={self.filled_qty}\n' \
               f'TriggerPrice={self.trigger_px}|TargetPrice={self.target_px}\n' \
               f'AmendPrice={self.amend_px}|AmendQuantity={self.amend_qty}'
    
    def __repr__(self):
        filled_size = self.filled_qty * self.side
        last_traded_size = self.ltq * self.side
        amend_size = self.amend_qty * self.side
        status_abbrev = self.get_status()
        return f'{self.creator}|{self.tv}|{self.acc}|{self.pdt}|{self.oid}|{self.eoid}|' \
               f'{self.type}|{self.tif}|{status_abbrev}|{self.size}@{self.price}|' \
               f'filled={filled_size}@{self.avg_px}|' \
               f'last={last_traded_size}@{self.ltp}|' \
               f'amend={amend_size}@{self.amend_px}|' \
               f'trigger={self.trigger_px}|target={self.target_px}|' \
               f'is_reduce_only={self.is_reduce_only}'
    
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, BaseOrder):
            return NotImplemented  # Allow other types to define equality with BaseOrder
        return self.key == other.key

    def __hash__(self) -> int:
        return hash(self.key)
