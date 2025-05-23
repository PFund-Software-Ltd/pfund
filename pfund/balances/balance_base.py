from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund.accounts.account_base import BaseAccount

import time
import logging
from decimal import Decimal
from dataclasses import dataclass, replace


class BaseBalance:
    @dataclass(frozen=True)
    class Balance:
        ts: float = 0.0
        wallet: Decimal = Decimal(0)
        available: Decimal = Decimal(0)
        margin: Decimal = Decimal(0)
        
    def __init__(self, account: BaseAccount, ccy: str):
        self.logger = logging.getLogger('balances')
        self.bkr = account.bkr
        self.account = account
        self.acc = account.acc
        # EXTEND: add Currency class (≈ Product), e.g. CryptoCurrency, FiatCurrency, ...
        # but need to find a way to add data/info/specs to each currency, e.g. region, country, ...
        self.ccy = ccy
        self._prev_balance = self.Balance()
        self._balance = self.Balance()

    def on_update(self, update, ts=None):
        update['ts'] = ts or time.time()
        self._prev_balance = self._balance
        self._balance = replace(self._balance, **update)
        if self._prev_balance != self._balance:
            self.logger.debug(f'{self}')

    @property
    def wallet(self):
        return self._balance.wallet

    @property
    def prev_wallet(self):
        return self._prev_balance.wallet

    @property
    def available(self):
        return self._balance.available

    @property
    def prev_available(self):
        return self._prev_balance.available

    @property
    def margin(self):
        return self._balance.margin

    @property
    def prev_margin(self):
        return self._prev_balance.margin
