from typing import NamedTuple, Literal

from decimal import Decimal


# used to tell if a timestamp is from the venue or created by pfund internally
class Timestamp(NamedTuple):
    value: float
    source: Literal["venue", "pfund"]


def trim_trailing_zeros(value: Decimal) -> Decimal:
    """Trims unnecessary trailing zeros from a Decimal without changing its value.

    e.g. 0.010 -> 0.01, 25000.00 -> 25000

    For whole numbers, `to_integral()` is used instead of `normalize()` to avoid
    the exponent form normalize() produces for integers (e.g. 25000 -> 2.5E+4).
    """
    integral = value.to_integral()
    return integral if value == integral else value.normalize()
