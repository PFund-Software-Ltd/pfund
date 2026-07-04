from enum import StrEnum


# NOTE: trailing stop orders are intentionally NOT treated as order types
# trailing is considered as a mechanism for stop orders in pfund
class OrderType(StrEnum):
    LIMIT = LMT = "LIMIT"
    MARKET = MKT = "MARKET"
    STOP_MARKET = STOP_MKT = STOP = "STOP_MARKET"
    STOP_LIMIT = STOP_LMT = "STOP_LIMIT"
