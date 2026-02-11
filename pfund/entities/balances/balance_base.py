from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund.typing import Currency

from decimal import Decimal
from dataclasses import dataclass, replace


class BaseBalance:
    @dataclass(frozen=True)
    class Snapshot:
        ts: float = 0.0
        wallet: Decimal = Decimal(0)
        available: Decimal = Decimal(0)
        margin: Decimal = Decimal(0)

    def __init__(self, ccy: Currency):
        self.ccy = ccy
        self._balance = self.Snapshot()

    def on_update(self, data: dict[str, Decimal], ts: float | None=None):
        self._balance = replace(self._balance, ts=ts, **data)

    @property
    def wallet(self) -> Decimal:
        return self._balance.wallet

    @property
    def available(self) -> Decimal:
        return self._balance.available

    @property
    def margin(self) -> Decimal:
        return self._balance.margin
