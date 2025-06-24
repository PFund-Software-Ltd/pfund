from enum import StrEnum


class BacktestMode(StrEnum):
    vectorized = 'vectorized'
    hybrid = 'hybrid'
    event_driven = 'event_driven'
