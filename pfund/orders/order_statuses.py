from enum import Enum


class MainOrderStatus(Enum):
    SUBMITTED = S = 1
    OPENED = O = 2
    CLOSED = C = 3
    REJECTED = R = -1
    MISSED = M = -2


class FillOrderStatus(Enum):
    PARTIAL = P = 1
    FILLED = F = 2


class CancelOrderStatus(Enum):
    SUBMITTED = S = 1
    CANCELLED = C = 2
    REJECTED = R = -1
    MISSED = M = -2


class AmendOrderStatus(Enum):
    SUBMITTED = S = 1
    AMENDED = A = 2
    REJECTED = R = -1
    MISSED = M = -2


# Indices in order status, 
# e.g. order status = 'S---', index 0 represents the MainOrderStatus etc.
STATUS_INDICES = {
    MainOrderStatus: 0,
    0: MainOrderStatus,
    FillOrderStatus: 1,
    1: FillOrderStatus,
    CancelOrderStatus: 2,
    2: CancelOrderStatus,
    AmendOrderStatus: 3,
    3: AmendOrderStatus
}