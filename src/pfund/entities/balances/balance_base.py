from __future__ import annotations
from typing import ClassVar

import time
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, computed_field

from pfund.typing import Currency


class BalanceSnapshot(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    updated_at: float = Field(default_factory=time.time)
    cash: Decimal | None = Field(default=None)  # wallet balance
    equity: Decimal | None = Field(default=None)  # ≈ margin balance
    available: Decimal | None = Field(default=None, ge=0)
    initial_margin: Decimal | None = Field(default=None)
    maintenance_margin: Decimal | None = Field(default=None)
    unrealized_pnl: Decimal | None = Field(default=None)
    realized_pnl: Decimal | None = Field(default=None)


class BaseBalance(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")
    _Snapshot: ClassVar[type[BalanceSnapshot]] = BalanceSnapshot
    _snapshot: BalanceSnapshot = PrivateAttr(default_factory=BalanceSnapshot)

    currency: Currency

    def on_snapshot_update(self, update: BalanceSnapshot) -> None:
        self._snapshot = self._Snapshot.model_validate(update)

    @property
    def ccy(self) -> Currency:
        return self.currency

    @computed_field
    @property
    def cash(self) -> Decimal:
        assert self._snapshot.cash is not None, "cash is None"
        return self._snapshot.cash

    @computed_field
    @property
    def equity(self) -> Decimal | None:
        return self._snapshot.equity

    @computed_field
    @property
    def available(self) -> Decimal | None:
        return self._snapshot.available

    @property
    def updated_at(self) -> float:
        return self._snapshot.updated_at

    def __str__(self):
        return f"Currency={self.currency} | Cash={self.cash} | Equity={self.equity} | Available={self.available}"

    def __repr__(self):
        return f"{self.currency} | cash={self.cash} | equity={self.equity} | available={self.available}"
