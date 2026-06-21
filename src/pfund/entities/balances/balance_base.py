from __future__ import annotations
from typing import ClassVar, TypedDict, TYPE_CHECKING, Any

if TYPE_CHECKING:

    class BalanceUpdate(TypedDict, total=False):
        total: Decimal
        available: Decimal


import time
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, computed_field


class BalanceSnapshot(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    updated_at: float = Field(default_factory=time.time)
    total: Decimal = Field(default=Decimal(0), ge=0)
    available: Decimal = Field(default=Decimal(0), ge=0)


class BaseBalance(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")
    _Snapshot: ClassVar[type[BalanceSnapshot]] = BalanceSnapshot
    _snapshot: BalanceSnapshot = PrivateAttr(default_factory=BalanceSnapshot)

    currency: str

    def on_snapshot_update(self, update: BalanceUpdate) -> None:
        self._snapshot = self._Snapshot.model_validate(update)

    @property
    def ccy(self) -> str:
        return self.currency

    @computed_field
    @property
    def total(self) -> Decimal:
        return self._snapshot.total

    @computed_field
    @property
    def available(self) -> Decimal:
        return self._snapshot.available

    @property
    def updated_at(self) -> float:
        return self._snapshot.updated_at

    def __str__(self):
        return f"Currency={self.currency} | Total={self.total} | Available={self.available}"

    def __repr__(self):
        return f"{self.currency} | total={self.total} | available={self.available}"
