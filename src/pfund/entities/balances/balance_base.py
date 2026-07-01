from __future__ import annotations
from typing import ClassVar, Literal, Generic, TypeVar, TypeAlias

from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, computed_field

from pfund.typing import AccountName, Currency


SnapshotT = TypeVar("SnapshotT", bound="BaseBalance.Snapshot")
BalanceUpdateSource: TypeAlias = Literal["get_balances", "websocket"]


class BalanceUpdate(BaseModel, Generic[SnapshotT]):
    ts: float = Field(description="Timestamp provided by the venue")
    account: AccountName
    snapshots: dict[Currency, SnapshotT]
    # the account-level consolidated balance (in the account's settlement
    # currency), distinct from the per-currency ``snapshots`` above
    account_balance: SnapshotT | None = None
    source: BalanceUpdateSource


class BaseBalance(BaseModel):
    class Snapshot(BaseModel):
        model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

        updated_at: float | None = Field(
            default=None,
            description="Timestamp provided by the venue, it is the same as the ``ts`` field in the BalanceUpdate",
        )
        cash: Decimal | None = Field(default=None)  # wallet balance
        equity: Decimal | None = Field(default=None)  # ≈ margin balance
        available: Decimal | None = Field(default=None, ge=0)
        initial_margin: Decimal | None = Field(default=None)
        maintenance_margin: Decimal | None = Field(default=None)
        unrealized_pnl: Decimal | None = Field(default=None)
        realized_pnl: Decimal | None = Field(default=None)

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")
    _snapshot: Snapshot = PrivateAttr(default_factory=Snapshot)

    currency: Currency

    def on_update(self, update: Snapshot) -> None:
        self._snapshot = self.Snapshot.model_validate(update)

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
    def updated_at(self) -> float | None:
        return self._snapshot.updated_at

    def __str__(self):
        return f"Currency={self.currency} | Cash={self.cash} | Equity={self.equity} | Available={self.available}"

    def __repr__(self):
        return f"{self.currency} | cash={self.cash} | equity={self.equity} | available={self.available}"
