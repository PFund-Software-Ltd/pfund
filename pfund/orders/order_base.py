import time
import logging
import hashlib
import math
from uuid import uuid4
from decimal import Decimal

from typing import Literal

from pfund.orders.order_statuses import *
from pfund.products.product_base import BaseProduct
from pfund.accounts.account_base import BaseAccount


class BaseOrder:
    def __init__(
            self, 
            account: BaseAccount,
            product: BaseProduct,
            side: int | str,
            qty: float | str,
            px=None,
            trigger_px=None,
            target_px=None,  # used for slippage calculation
            o_type='LIMIT',
            tif='GTC',
            is_reduce_only=False,
            oid='',
            remark='',  # user's remark, e.g. reason why placing this order
            **kwargs
        ):
        self.logger = logging.getLogger('orders')
        self.strat = account.strat
        self.acc = account.acc
        self.account = account
        self.bkr = product.bkr
        self.exch = product.exch
        self.tv = self.trading_venue = self.exch if self.bkr == 'CRYPTO' else self.bkr
        self.pdt = product.pdt
        self.product = product

        if type(side) is str:
            side = side.upper()
            assert side in ['BUY', 'SELL']
            self.side = 1 if side == 'BUY' else -1
        else:
            assert side in [1, -1]
            self.side = side
        
        assert qty
        self.qty = self._adjust_by_lot_size(qty)
        self.prev_qty = self.remain_qty = self.qty
        self.size = self.qty * self.side
        self.filled_qty = self.last_traded_qty = self.ltq = 0.0
        self.amend_qty = None

        self.px = self._adjust_by_tick_size(px) if px else None
        self.prev_px = self.px
        self.trigger_px = self._adjust_by_tick_size(trigger_px) if trigger_px else None
        self.target_px = self._adjust_by_tick_size(target_px) if target_px else None
        self.avg_px = self.last_traded_px = self.ltp = None
        self.amend_px = None
        
        self.type = self.order_type = o_type
        self.tif = self.time_in_force = tif
        self.is_reduce_only = is_reduce_only

        self._oid = oid if oid else self._generate_oid()
        self._eoid = ''

        self._status = [None] * 4
        self._status_reasons = {}  # { MainOrderStatus: reason}
        self.timestamps = {}  # { MainOrderStatus.SUBMITTED: ts }
        self.trades = []

        self.remark = remark

        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def oid(self):
        return self._oid

    @property
    def eoid(self):
        return self._eoid
    
    @eoid.setter
    def eoid(self, eoid):
        if eoid and not self._eoid:
            self._eoid = eoid

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
            return self.qtyQ * -self.side

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
            if not self.px:
                return Decimal(0) if self.remain_qty == 0 else None
            else:
                return self.px * self.remain_qty
    
    def is_inverse(self):
        if hasattr(self.product, 'is_inverse'):
            return self.product.is_inverse()
        else:
            return False
        
    def _generate_oid(self):
        # return hashlib.md5((self.strat + str(time.time())).encode()).hexdigest()
        return uuid4()
    
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
        status_index = STATUS_INDICES[AmendOrderStatus]
        return self._status[status_index] == AmendOrderStatus.SUBMITTED

    def is_cancelling(self):
        status_index = STATUS_INDICES[CancelOrderStatus]
        return self._status[status_index] == CancelOrderStatus.SUBMITTED

    def is_opened(self):
        status_index = STATUS_INDICES[MainOrderStatus]
        return self._status[status_index] == MainOrderStatus.OPENED

    def is_closed(self):
        return not self.is_opened()
    
    def is_traded(self):
        return self.filled_qty > 0
    
    def is_filled(self):
        return self.filled_qty == self.qty

    def on_status_update(self, status, ts=None, reason='') -> bool:
        is_updated = False
        prev_status = self._status
        status_type = type(status)
        status_index = STATUS_INDICES[status_type]
        self._status[status_index] = status
        self._status_reasons[status_type] = reason
        self.timestamps[status] = ts or time.time()
        if prev_status != self._status:
            is_updated = True
            self.logger.debug(repr(self))
        return is_updated

    def on_trade_update(self, avg_px, filled_qty, last_traded_px, last_traded_qty) -> bool:
        prev_avg_px = self.avg_px if self.avg_px else 0.0
        prev_filled_qty = self.filled_qty
        is_updated = False
        if not filled_qty:
            self.logger.error(f'trade update has {filled_qty=} {self.strat=} {self.bkr} {self.exch} {self.oid=}')
        else:
            if filled_qty > prev_filled_qty:
                is_updated = True
                self.filled_qty = filled_qty
                self.last_traded_qty = self.ltq = filled_qty - prev_filled_qty
                self.remain_qty = self.qty - filled_qty
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
                    self.avg_px = self.ltp = self.px
                self.trades.append({'px': self.ltp, 'qty': self.ltq})
            elif filled_qty < prev_filled_qty:
                self.logger.warning(f'Delayed trade msg {self.strat=} {self.bkr} {self.exch} {self.oid=} ({filled_qty=} < {prev_filled_qty=})')
        return is_updated          

    def __str__(self):
        side_str = 'BUY' if self.side == 1 else 'SELL'
        return f'Strategy={self.strat}|Broker={self.bkr}|Exchange={self.exch}|Account={self.acc}|Product={self.pdt}\n' \
               f'OrderType={self.type}|TimeInForce={self.tif}|IsReduceOnly={self.is_reduce_only}\n' \
               f'Side={side_str}|Price={self.px}|Quantity={self.qty}\n' \
               f'AveragePrice={self.avg_px}|FilledQuantity={self.filled_qty}\n' \
               f'TriggerPrice={self.trigger_px}|TargetPrice={self.target_px}\n' \
               f'AmendPrice={self.amend_px}|AmendQuantity={self.amend_qty}'
    
    def __repr__(self):
        filled_size = self.filled_qty * self.side
        last_traded_size = self.ltq * self.side
        amend_size = self.amend_qty * self.side
        status_abbrev = self.get_status()
        return f'{self.strat}|{self.tv}|{self.acc}|{self.pdt}|{self.oid}|{self.eoid}|' \
               f'{self.type}|{self.tif}|{status_abbrev}|{self.size}@{self.px}|' \
               f'filled={filled_size}@{self.avg_px}|' \
               f'last={last_traded_size}@{self.ltp}|' \
               f'amend={amend_size}@{self.amend_px}|' \
               f'trigger={self.trigger_px}|target={self.target_px}|' \
               f'is_reduce_only={self.is_reduce_only}'

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
        print(f'Order Status(oid={self._oid}): {readable_o_status}')
