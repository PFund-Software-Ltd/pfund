import time
from decimal import Decimal
from dataclasses import dataclass, replace

from numpy import sign

from pfund.positions.position_base import BasePosition


class IBPosition(BasePosition):
    # TODO, consider create IBStockPosition etc. to separate products
    @dataclass(frozen=True)
    class Position:
        ts: float = 0.0
        size: Decimal = Decimal(0)
        avg_px: Decimal = Decimal(0)
        market_px: Decimal = Decimal(0)
        liquidation_px: Decimal = Decimal(0)
        unrealized_pnl: Decimal = Decimal(0)
        realized_pnl: Decimal = Decimal(0)

    def __init__(self, account, product):
        super().__init__(account, product)
        # if true, it is a virtual position, e.g. USD_CAD_CASH = +1 (bought 1 USD and sold `x` CAD)
        self._is_virtual = True if product.ptype in ['CRYPTO', 'CASH'] else False
        if product.is_asset() and not self._is_virtual:
            self._is_security = True
        else:
            self._is_security = False
        self._position = self.Position()

    def is_security(self):
        return self._is_security 

    # TODO
    def on_update(self, update, ts=None):
        now = time.time()
        update['ts'] = ts or now
        self._position = replace(self._position, **update)

    # TODO
    def on_update_by_trade(self, o):
        lts = o.ltq * o.side
        ttl_size = self.size + lts
        new_side = sign(ttl_size)
        ttl_qty = abs(ttl_size)
        # calc avg_px
        if new_side != 0:
            if o.side != self.side:
                if o.ltq > self.qty:
                    avg_px = o.ltp
                else:
                    avg_px = self.avg_px
            else:
                avg_px = self.avg_px
                qty = self.qty
                avg_px = (avg_px * qty + o.ltp * o.ltq) / ttl_qty
        else:
            avg_px = Decimal(0)
        update = {'size': ttl_size, 'avg_px': avg_px, 'liquidation_px': Decimal(0), 'unrealized_pnl': Decimal(0), 'realized_pnl': Decimal(0)}
        # FIXME
        # ts = o.ltt
        ts = time.time()
        self.logger.debug(f'update {self.exch} {self.pdt} {update} by oid {o.oid} ({lts}@{o.ltp}) ts={ts}')
        self.on_update(update, ts=ts)
        