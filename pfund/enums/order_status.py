from enum import StrEnum


class MainOrderStatus(StrEnum):
    SUBMITTED = S = 'SUBMITTED'
    OPENED = O = 'OPENED'
    CLOSED = C = 'CLOSED'
    REJECTED = R = 'REJECTED'
    MISSED = M = 'MISSED'


class FillOrderStatus(StrEnum):
    PARTIAL = P = 'PARTIAL'
    FILLED = F = 'FILLED'


class CancelOrderStatus(StrEnum):
    SUBMITTED = S = 'SUBMITTED'
    CANCELLED = C = 'CANCELLED'
    REJECTED = R = 'REJECTED'
    MISSED = M = 'MISSED'


class AmendOrderStatus(StrEnum):
    SUBMITTED = S = 'SUBMITTED'
    AMENDED = A = 'AMENDED'
    REJECTED = R = 'REJECTED'
    MISSED = M = 'MISSED'
