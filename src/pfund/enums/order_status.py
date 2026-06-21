from enum import StrEnum, nonmember


class OrderStatus:
    class Main(StrEnum):
        position = nonmember(0)
        BLOCKED = B = "BLOCKED"  # blocked by internal checking, e.g. risk management
        SUBMITTED = S = "SUBMITTED"
        OPENED = O = "OPENED"
        CLOSED = C = "CLOSED"
        REJECTED = R = "REJECTED"
        MISSED = M = "MISSED"

    class Fill(StrEnum):
        position = nonmember(1)
        PARTIAL = P = "PARTIAL"
        FILLED = F = "FILLED"

    class Cancel(StrEnum):
        position = nonmember(2)
        SUBMITTED = S = "SUBMITTED"
        CANCELLED = C = "CANCELLED"
        REJECTED = R = "REJECTED"
        MISSED = M = "MISSED"

    class Amend(StrEnum):
        position = nonmember(3)
        SUBMITTED = S = "SUBMITTED"
        AMENDED = A = "AMENDED"
        REJECTED = R = "REJECTED"
        MISSED = M = "MISSED"

    # parts in slot order; index == each part's `position`
    PARTS = (Main, Fill, Cancel, Amend)
