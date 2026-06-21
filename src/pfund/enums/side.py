from typing import Any

from enum import IntEnum
from decimal import Decimal


class Side(IntEnum):
    BUY = LONG = 1
    SELL = SHORT = -1

    @classmethod
    def _missing_(cls, value: Any):
        if isinstance(value, (int, float, Decimal)):
            if value > 0:
                return cls.BUY
            elif value < 0:
                return cls.SELL
        return None

    def __str__(self) -> str:
        return self.name

    __repr__ = __str__
