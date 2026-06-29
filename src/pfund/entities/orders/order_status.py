from __future__ import annotations
from typing import TypeAlias

import time
from enum import StrEnum

from pfund.utils import Timestamp


class FIX(StrEnum):
    """The single canonical order status, i.e. FIX OrdStatus (tag 39).

    If an order could carry only one status, it would be this one. It is
    *derived* from the four orthogonal OrderStatus slots (see OrderStatus.FIX),
    never stored.

    Member values are the exact FIX wordings. The enumeration follows
    FIX 5.0 SP2 (published by the FIX Trading Community as "FIX Latest").
    The OrdStatus value set has been stable since FIX 4.x; it is pinned
    here so a future spec change is an explicit edit, not a silent drift.
    Codes in comments are the on-the-wire tag-39 values.
    """

    PENDING_NEW = "Pending New"  # A
    NEW = "New"  # 0
    PARTIALLY_FILLED = "Partially filled"  # 1
    FILLED = "Filled"  # 2
    DONE_FOR_DAY = "Done for day"  # 3
    CANCELED = "Canceled"  # 4  (FIX uses the US spelling, one 'l')
    REPLACED = "Replaced"  # 5  (deprecated in FIX 5.0 SP2)
    PENDING_CANCEL = "Pending Cancel"  # 6
    STOPPED = "Stopped"  # 7
    REJECTED = "Rejected"  # 8
    SUSPENDED = "Suspended"  # 9
    CALCULATED = "Calculated"  # B
    EXPIRED = "Expired"  # C
    ACCEPTED_FOR_BIDDING = "Accepted for Bidding"  # D
    PENDING_REPLACE = "Pending Replace"  # E


class OrderStatus:
    class Main(StrEnum):
        BLOCKED = B = "BLOCKED"  # blocked by internal checking, e.g. risk management
        SUBMITTED = S = "SUBMITTED"  # pfund sent the order; no venue response yet
        PENDING = P = (
            "PENDING"  # venue acknowledged receipt, not yet working (FIX Pending New)
        )
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
        SUBMITTED = S = (
            "SUBMITTED"  # pfund sent the cancel request; no venue response yet
        )
        PENDING = P = (
            "PENDING"  # venue acknowledged the cancel, not yet done (FIX Pending Cancel)
        )
        CANCELLED = C = "CANCELLED"
        REJECTED = R = "REJECTED"
        MISSED = M = "MISSED"

    class Amend(StrEnum):
        SUBMITTED = S = (
            "SUBMITTED"  # pfund sent the amend request; no venue response yet
        )
        PENDING = P = (
            "PENDING"  # venue acknowledged the amend, not yet done (FIX Pending Replace)
        )
        AMENDED = A = "AMENDED"
        REJECTED = R = "REJECTED"
        MISSED = M = "MISSED"

    _CATEGORIES = (Main, Fill, Cancel, Amend)
    Category: TypeAlias = Main | Fill | Cancel | Amend

    # Maps each internal slot value to its FIX OrdStatus. Nested by category
    # because slot values collide across categories (e.g. Cancel.SUBMITTED ==
    # Amend.SUBMITTED as StrEnums), so a flat {member: FIX} dict would merge them.
    # Slot values absent here have no FIX equivalent and are ignored by FIX:
    #   - *.SUBMITTED                 -> pfund-local "request sent"; FIX has no
    #                                    status before the venue's first report,
    #                                    so only the venue-reported *.PENDING maps
    #   - Main.BLOCKED / Main.MISSED  -> pfund-internal, never sent / no venue ack
    #   - Main.CLOSED                 -> generic terminal; the real outcome lives
    #                                    in the Fill/Cancel slot (FILLED/CANCELED)
    #   - Cancel/Amend.REJECTED|MISSED-> the request failed; the order itself is
    #                                    still working, so it must not override
    #   - Amend.AMENDED               -> order is working again; NEW vs
    #                                    PARTIALLY_FILLED comes from Main/Fill
    _FIX_MAPPING: dict[type[OrderStatus.Category], dict[OrderStatus.Category, FIX]] = {
        Main: {
            Main.PENDING: FIX.PENDING_NEW,
            Main.ACTIVE: FIX.NEW,
            Main.REJECTED: FIX.REJECTED,
        },
        Fill: {
            Fill.PARTIAL: FIX.PARTIALLY_FILLED,
            Fill.FILLED: FIX.FILLED,
        },
        Cancel: {
            Cancel.PENDING: FIX.PENDING_CANCEL,
            Cancel.CANCELLED: FIX.CANCELED,
        },
        Amend: {
            Amend.PENDING: FIX.PENDING_REPLACE,
        },
    }

    # When several slots map to a FIX status at once, the highest-precedence one
    # wins: terminal outcomes first, then in-flight requests, then working states.
    _FIX_PRECEDENCE: tuple[FIX, ...] = (
        FIX.FILLED,
        FIX.CANCELED,
        FIX.REJECTED,
        FIX.PENDING_CANCEL,
        FIX.PENDING_REPLACE,
        FIX.PENDING_NEW,
        FIX.PARTIALLY_FILLED,
        FIX.NEW,
    )

    def __init__(self):
        self.main: OrderStatus.Main | None = None
        self.fill: OrderStatus.Fill | None = None
        self.cancel: OrderStatus.Cancel | None = None
        self.amend: OrderStatus.Amend | None = None
        self._reasons: dict[type[OrderStatus.Category], str] = {}
        self._updated_ats: dict[type[OrderStatus.Category], Timestamp] = {}

    def is_submitted(self) -> bool:
        return self.main == OrderStatus.Main.SUBMITTED

    def is_pending(self) -> bool:
        return self.main == OrderStatus.Main.PENDING

    def is_active(self) -> bool:
        return self.main == OrderStatus.Main.ACTIVE

    def is_closed(self) -> bool:
        return self.main == OrderStatus.Main.CLOSED

    def is_cancelling(self, include_submitted: bool = False) -> bool:
        # PENDING = venue-acked cancel in flight; include_submitted also counts
        # the pfund-local "request sent" state (SUBMITTED) as cancelling.
        if self.cancel == OrderStatus.Cancel.PENDING:
            return True
        return include_submitted and self.cancel == OrderStatus.Cancel.SUBMITTED

    def is_cancelled(self) -> bool:
        return self.cancel == OrderStatus.Cancel.CANCELLED

    def is_amending(self, include_submitted: bool = False) -> bool:
        # PENDING = venue-acked amend in flight; include_submitted also counts
        # the pfund-local "request sent" state (SUBMITTED) as amending.
        if self.amend == OrderStatus.Amend.PENDING:
            return True
        return include_submitted and self.amend == OrderStatus.Amend.SUBMITTED

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
    def FIX(self) -> FIX | None:
        """Collapse the four orthogonal slots into the single canonical FIX
        OrdStatus (tag 39). Returns None for purely pfund-internal states that
        have no FIX equivalent (e.g. only Main.BLOCKED or Main.MISSED is set).
        """
        candidates = {
            self._FIX_MAPPING[type(s)][s]
            for s in self._slots
            if s is not None and s in self._FIX_MAPPING[type(s)]
        }
        return next(
            (status for status in self._FIX_PRECEDENCE if status in candidates), None
        )

    @property
    def _slots(self) -> tuple[OrderStatus.Category | None, ...]:
        return tuple(getattr(self, c.__name__.lower()) for c in self._CATEGORIES)

    def __str__(self):
        return " | ".join(s.name if s else "----" for s in self._slots)

    def __repr__(self):
        return "".join(s.name[0] if s else "-" for s in self._slots)
