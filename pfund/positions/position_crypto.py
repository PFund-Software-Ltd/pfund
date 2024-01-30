import time
from decimal import Decimal
from dataclasses import dataclass, replace

from numpy import sign

from pfund.positions.position_base import BasePosition


class CryptoPosition(BasePosition):
    @dataclass(frozen=True)
    class Position:
        ts: float = 0.0
        size: Decimal = Decimal(0)
        avg_px: Decimal = Decimal(0)
        liquidation_px: Decimal = Decimal(0)
        unrealized_pnl: Decimal = Decimal(0)
        realized_pnl: Decimal = Decimal(0)

    def __init__(self, account, product):
        super().__init__(account, product)
        self.multi = product.multi
        self._long_position = self.Position()
        self._short_position = self.Position()

    def is_inverse(self):
        return self.product.is_inverse()

    def on_update(self, update, ts=None):
        now = time.time()
        for side in update:
            update_per_side = update[side]
            update_per_side['ts'] = ts or now
            update_per_side['size'] = side * update_per_side['qty']
            del update_per_side['qty']
            if side == 1:
                self._long_position = replace(self._long_position, **update_per_side)
            elif side == -1:
                self._short_position = replace(self._short_position, **update_per_side)
            elif side == 0:
                self._long_position = self.Position()
                self._short_position = self.Position()
        self._combine_long_and_short_positions()

    # TODO
    def on_update_by_trade(self, o):
        ttl_size = self.size + o.ltz
        new_side = sign(ttl_size)
        ttl_qty = abs(ttl_size)
        # calc avg_px
        if new_side != 0:
            if o.side != self.side:
                if o.ltq > self.qty:
                    avg_px = o.ltp
                else:
                    avg_px = self.long_avg_px if self.side == 1 else self.short_avg_px
            else:
                avg_px = self.long_avg_px if self.side == 1 else self.short_avg_px
                qty = self.long_qty if self.side == 1 else self.short_qty
                avg_px = (avg_px * qty + o.ltp * o.ltq) / ttl_qty
        else:
            avg_px = Decimal(0)
        update = {new_side: {'qty': ttl_qty, 'avg_px': avg_px, 'liquidation_px': Decimal(0), 'unrealized_pnl': Decimal(0), 'realized_pnl': Decimal(0)}}
        ts = o.ltt
        self.logger.debug(f'update {self.exch} {self.pdt} {update} by oid {o.oid} ({o.lts}@{o.ltp}) ts={ts}')
        self.on_update(update)

    def _combine_long_and_short_positions(self):
        size = self.long_qty - self.short_qty
        side = sign(size)
        if side == 1:       
            avg_px = self.long_avg_px
        elif side == -1:
            avg_px = self.short_avg_px
        elif side == 0:
            avg_px = Decimal(0)
        update = {'size': size, 'avg_px': avg_px}
        self._prev_position = self._position
        self._position = replace(self._position, **update)
        if self._prev_position != self._position:
            self.logger.debug(f'{self}')

    @property
    def linear_size(self):
        if self.is_empty():
            return Decimal(0)
        else:
            return self.size * self.multi / (self.avg_px if self.is_inverse() else 1)

    @property
    def linear_qty(self):
        return abs(self.linear_size)

    @property
    def long_qty(self):
        return abs(self._long_position.size)

    @property
    def short_qty(self):
        return abs(self._short_position.size)

    @property
    def long_avg_px(self):
        return self._long_position.avg_px

    @property
    def short_avg_px(self):
        return self._short_position.avg_px