from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund.accounts.account_base import BaseAccount
    from pfund.products.product_base import BaseProduct

import logging

from numpy import sign


class BasePosition:
    def __init__(self, account: BaseAccount, product: BaseProduct):
        self.logger = logging.getLogger('positions')
        self.account = account
        self.product = product
        self.bkr = product.bkr
        self.exch = product.exch
        self.acc = account.acc
        self.strat = account.strat
        self.pdt = product.name
        self.pair, self.ptype = product.pair, product.ptype
        self.bccy, self.qccy = product.bccy, product.qccy
        self._prev_position = self.Position()
        self._position = self.Position()

    def is_empty(self):
        return not self.side
    
    @property
    def side(self):
        return sign(self.size)

    @property
    def quantity(self):
        return abs(self.size)
    qty = quantity

    @property
    def size(self):
        return self._position.size

    @property
    def prev_size(self):
        return self._prev_position.size
    
    @property
    def avg_px(self):
        return self._position.avg_px
    average_price = avg_px

    def __str__(self):
        return f'Broker={self.bkr}|Exchange={self.exch}|Account={self.acc}|Product={self.pdt}|Position={self._position}'

    def __repr__(self):
        return f'{self.bkr}:{self.exch}:{self.acc}:{self.pdt}:{self._position}'
