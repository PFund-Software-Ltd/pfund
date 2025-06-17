from enum import StrEnum


# EXTEND
class OrderType(StrEnum):
    LIMIT = LMT = 'LIMIT'
    MARKET = MKT = 'MARKET'
    STOP_MARKET = STOP_MKT = STOP = 'STOP_MARKET'
    STOP_LIMIT = STOP_LMT = 'STOP_LIMIT'
    # TRAILING_STOP = 'TRAILING_STOP'
    # OCO = 'OCO'
    # OTO = 'OTO'