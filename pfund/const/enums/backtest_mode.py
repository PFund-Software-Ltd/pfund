from enum import StrEnum


class BacktestMode(StrEnum):
    vectorized = 'vectorized'
    event_driven = 'event_driven'
