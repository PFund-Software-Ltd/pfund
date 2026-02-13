from enum import StrEnum


class BacktestMode(StrEnum):
    VECTORIZED = 'VECTORIZED'
    HYBRID = 'HYBRID'
    EVENT_DRIVEN = 'EVENT_DRIVEN'
