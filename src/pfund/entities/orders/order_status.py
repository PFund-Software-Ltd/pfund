from __future__ import annotations
from typing import TypeAlias

import time
from enum import StrEnum

from pfund.utils import Timestamp


class OrderStatus:
    class Main(StrEnum):
        BLOCKED = B = "BLOCKED"  # blocked by internal checking, e.g. risk management
        SUBMITTED = S = "SUBMITTED"
        ACTIVE = A = "ACTIVE"
        CLOSED = C = "CLOSED"
        REJECTED = R = "REJECTED"  # rejected by the external trading venue
        MISSED = M = (
            "MISSED"  # missed acknowledgement, i.e. no response from the external trading venue
        )

    class Fill(StrEnum):
        PARTIAL = P = "PARTIAL"
        FILLED = F = "FILLED"

    class Cancel(StrEnum):
        SUBMITTED = S = "SUBMITTED"
        CANCELLED = C = "CANCELLED"
        REJECTED = R = "REJECTED"
        MISSED = M = "MISSED"

    class Amend(StrEnum):
        SUBMITTED = S = "SUBMITTED"
        AMENDED = A = "AMENDED"
        REJECTED = R = "REJECTED"
        MISSED = M = "MISSED"

    _CATEGORIES = (Main, Fill, Cancel, Amend)
    Category: TypeAlias = Main | Fill | Cancel | Amend

    def __init__(self):
        self.main: OrderStatus.Main | None = None
        self.fill: OrderStatus.Fill | None = None
        self.cancel: OrderStatus.Cancel | None = None
        self.amend: OrderStatus.Amend | None = None
        self._reasons: dict[type[OrderStatus.Category], str] = {}
        self._updated_ats: dict[type[OrderStatus.Category], Timestamp] = {}

    def is_submitted(self) -> bool:
        return self.main == OrderStatus.Main.SUBMITTED

    def is_active(self) -> bool:
        return self.main == OrderStatus.Main.ACTIVE

    def is_closed(self) -> bool:
        return self.main == OrderStatus.Main.CLOSED

    def is_cancelling(self) -> bool:
        return self.cancel == OrderStatus.Cancel.SUBMITTED

    def is_cancelled(self) -> bool:
        return self.cancel == OrderStatus.Cancel.CANCELLED

    def is_amending(self) -> bool:
        return self.amend == OrderStatus.Amend.SUBMITTED

    def is_amended(self) -> bool:
        return self.amend == OrderStatus.Amend.AMENDED

    def update(
        self, status: OrderStatus.Category, ts: float | None = None, reason: str = ""
    ) -> bool:
        is_updated = False
        category = type(status)  # Main / Fill / Cancel / Amend
        if category not in self._CATEGORIES:
            raise ValueError(f"Invalid order status: {status!r}")
        slot = category.__name__.lower()  # Main -> "main", Fill -> "fill", ...
        if getattr(self, slot) == status:  # idempotent: same value = no transition
            return is_updated
        setattr(self, slot, status)
        if ts:
            self._updated_ats[category] = Timestamp(ts, "venue")
        else:
            self._updated_ats[category] = Timestamp(time.time(), "pfund")
        self._reasons[category] = reason
        is_updated = True
        return is_updated

    @property
    def _slots(self) -> tuple[OrderStatus.Category | None, ...]:
        return tuple(getattr(self, c.__name__.lower()) for c in self._CATEGORIES)

    def __str__(self):
        return " | ".join(s.name if s else "----" for s in self._slots)

    def __repr__(self):
        return "".join(s.name[0] if s else "-" for s in self._slots)
