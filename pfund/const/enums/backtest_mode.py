from enum import StrEnum


class BacktestMode(StrEnum):
    VECTORIZED = 'VECTORIZED'
    EVENT_DRIVEN = 'EVENT_DRIVEN'
