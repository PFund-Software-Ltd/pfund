from __future__ import annotations
from typing import ClassVar, cast
from typing_extensions import override

import re
from enum import StrEnum


class ResolutionUnit(StrEnum):
    # MEANING = canonical value, (aliases, ...)
    YEAR = 'y', ('yr', 'year', 'years')
    MONTH = 'mo', ('mon', 'mons', 'month', 'months')
    WEEK = 'w', ('wk', 'week', 'weeks')
    DAY = 'd', ('day', 'days')
    HOUR = 'h', ('hour', 'hours')
    MINUTE = 'm', ('min', 'mins', 'minute', 'minutes')
    SECOND = 's', ('sec', 'secs', 'second', 'seconds')
    TICK = 't', ('tick', 'ticks')
    QUOTE = 'q', ('quote', 'quotes')

    _aliases: tuple[str, ...]

    def __new__(cls, value: str, aliases: tuple[str, ...] = ()):
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj._aliases = (value,) + aliases  # include canonical value
        return obj

    @property
    def aliases(self) -> tuple[str, ...]:
        return self._aliases

    @classmethod
    @override
    def _missing_(cls, value: str) -> ResolutionUnit | None:
        '''Called when lookup fails - search by alias'''
        value_lower = value.lower()
        for unit in cls:
            if value_lower in unit._aliases:
                return unit
        return None  # Let default error handling take over

    @classmethod
    def pattern(cls) -> str:
        """Generate regex pattern matching any alias for all units."""
        all_aliases = []
        for unit in cls:
            all_aliases.extend(unit._aliases)
        return '|'.join(all_aliases)


class Resolution:
    DEFAULT_ORDERBOOK_LEVEL: ClassVar[int] = 1

    def __init__(self, resolution: Resolution | str):
        """
        Args:
            resolution: e.g. '1m', '1minute', '1quote_L1'
            If the input is a data_type (e.g. 'minute', 'daily'),
            it will be converted to resolution by adding '1' to the beginning.
            e.g. 'minute' -> '1m', 'daily' -> '1d'
        """
        from pfund.datas.timeframe import Timeframe, TimeframeStr

        if isinstance(resolution, Resolution):
            # Copy all attributes from the resolution object
            self.__dict__.update(resolution.__dict__)
            return

        period, timeframe, orderbook_level = self._parse(resolution)
        self.period: int = int(period)
        self.unit: ResolutionUnit = ResolutionUnit(timeframe)
        self.timeframe: Timeframe = Timeframe(cast(TimeframeStr, self.unit.value))
        self.orderbook_level: int | None = self._resolve_orderbook_level(orderbook_level)

    def _parse(self, resolution: str) -> tuple[str, str, str | None]:
        """Parse resolution string and extract period, timeframe, and orderbook level.
        Returns:
            tuple[str, str, str | None]: period string, timeframe string, orderbook level string
        """
        # Reject invalid characters before transformation
        assert not resolution.strip().startswith('-'), f"Invalid {resolution=}, period cannot be negative"
        
        # Add "1" if the resolution doesn't start with a number
        if not re.match(r"^\d", resolution):
            resolution = "1" + resolution

        # Only remove hyphens and underscores after the initial numbers
        resolution = re.sub(r"^(\d+)[-_]", r"\1", resolution)
        
        # validate resolution pattern
        assert re.match(rf"^[1-9]\d*({ResolutionUnit.pattern()})(?:_L[1-3])?$", resolution, re.IGNORECASE), (
            f"Invalid {resolution=}, pattern should be e.g. '1d', '2m', '3h', '1quote_L1' etc."
        )
        
        # extract orderbook level if it exists
        resolution, *orderbook_level = resolution.strip().split("_")
        if not orderbook_level:
            orderbook_level = None
        else:
            orderbook_level = orderbook_level[0]

        # extract period and timeframe
        period, timeframe = re.split(r"(\d+)", resolution.strip())[1:]
        return period, timeframe, orderbook_level
    
    def _resolve_orderbook_level(self, orderbook_level: str | None) -> int | None:
        if self.is_quote():
            if orderbook_level:
                orderbook_level = int(orderbook_level[0][-1])
            else:
                orderbook_level = self.DEFAULT_ORDERBOOK_LEVEL
        else:
            orderbook_level = None
        return orderbook_level
    
    def to_seconds(self) -> int:
        assert self.is_bar(), f"{self!r} is not a bar resolution"
        return self._value()

    def _value(self) -> int:
        """lower value = higher resolution"""
        return self.period * self.timeframe.unit.value * (self.orderbook_level or 1)

    def is_quote_l1(self):
        return self.is_quote() and self.orderbook_level == 1

    def is_quote_l2(self):
        return self.is_quote() and self.orderbook_level == 2

    def is_quote_l3(self):
        return self.is_quote() and self.orderbook_level == 3

    def is_quote(self):
        return self.timeframe.is_quote()

    def is_tick(self):
        return self.timeframe.is_tick()

    def is_bar(self):
        return self.is_second() or self.is_minute() or self.is_hour() or self.is_day()

    def is_second(self):
        return self.timeframe.is_second()

    def is_minute(self):
        return self.timeframe.is_minute()

    def is_hour(self):
        return self.timeframe.is_hour()

    def is_day(self):
        return self.timeframe.is_day()
    
    def is_week(self):
        return self.timeframe.is_week()
    
    def is_month(self):
        return self.timeframe.is_month()
    
    def is_year(self):
        return self.timeframe.is_year()

    def higher(self) -> Resolution:
        """Rotate to the next higher resolution. e.g. 1m > 1h, higher resolution = lower timeframe"""
        if self.is_quote():
            if self.orderbook_level < 3:
                return Resolution(
                    "1" + repr(self.timeframe) + "_L" + str(self.orderbook_level + 1)
                )
            else:
                return self
        else:
            return Resolution("1" + repr(self.timeframe.lower()))

    def lower(self) -> Resolution:
        """Rotate to the next lower resolution. e.g. 1h < 1m, lower resolution = higher timeframe"""
        if self.is_quote() and self.orderbook_level > 1:
            return Resolution(
                "1" + repr(self.timeframe) + "_L" + str(self.orderbook_level - 1)
            )
        else:
            return Resolution("1" + repr(self.timeframe.higher()))

    def to_unit(self) -> Resolution:
        """Convert to unit resolution. e.g. 5m -> 1m"""
        timeframe = repr(self.timeframe)
        if self.is_quote():
            return Resolution("1" + timeframe + "_L" + str(self.orderbook_level))
        else:
            return Resolution("1" + timeframe)

    def get_higher_resolutions(
        self, ignore_period: bool = False, exclude_quote: bool = False
    ) -> list[Resolution]:
        """Get all resolutions with higher granularity (finer time intervals) than this one.

        Args:
            ignore_period: If False and this resolution has a period > 1 (e.g., 5m),
                the unit resolution (e.g., 1m) is included as a higher resolution.
                If True, only resolutions of different time units are considered.
            exclude_quote: If True, stop before including quote/tick resolution.

        Returns:
            List of resolutions with higher granularity, ordered from closest to finest.
        """
        higher_resolutions: list[Resolution] = []
        resolution = self
        unit_resolution = self.to_unit()
        if not ignore_period and resolution != unit_resolution:
            higher_resolutions.append(unit_resolution)
        while (higher_resolution := unit_resolution.higher()) != unit_resolution:
            unit_resolution = higher_resolution
            if higher_resolution.is_quote() and exclude_quote:
                break
            higher_resolutions.append(higher_resolution)
        return higher_resolutions

    def get_lower_resolutions(self, exclude_quote: bool = False) -> list[Resolution]:
        """Get all resolutions with lower granularity (coarser time intervals) than this one.

        Args:
            exclude_quote: If True, skip quote/tick resolutions in the result
                (they are still traversed but not included).

        Returns:
            List of resolutions with lower granularity, ordered from closest to coarsest.
        """
        lower_resolutions: list[Resolution] = []
        resolution = self
        while (lower_resolution := resolution.lower()) != resolution:
            resolution = lower_resolution
            if lower_resolution.is_quote() and exclude_quote:
                continue
            lower_resolutions.append(lower_resolution)
        return lower_resolutions

    def __str__(self):
        resolution = f"{self.period}_{self.unit.name}"
        if self.orderbook_level:
            resolution += f"_L{self.orderbook_level}"
        return resolution

    def __repr__(self):
        resoultion = f"{self.period}{self.unit.value}"
        if self.orderbook_level:
            resoultion += f"_L{self.orderbook_level}"
        return resoultion

    def __hash__(self):
        return self._value()

    def is_strict_equal(self, other):
        """1h = 60m when using __eq__ to compare resolutions, but in strict_equal, 1h != 60m"""
        return self._value() == other._value() and self.timeframe == other.timeframe

    def __eq__(self, other):
        if not isinstance(other, Resolution):
            return NotImplemented
        return self._value() == other._value()

    def __ne__(self, other):
        if not isinstance(other, Resolution):
            return NotImplemented
        return not self == other

    # NOTE: higher value = lower resolution and vice versa
    # e.g. 1m (higher resolution, value=60) > 1h (lower resolution, value=3600)
    def __lt__(self, other):
        if not isinstance(other, Resolution):
            return NotImplemented
        return self._value() > other._value()

    def __le__(self, other):
        if not isinstance(other, Resolution):
            return NotImplemented
        return self._value() >= other._value()

    def __gt__(self, other):
        if not isinstance(other, Resolution):
            return NotImplemented
        return self._value() < other._value()

    def __ge__(self, other):
        if not isinstance(other, Resolution):
            return NotImplemented
        return self._value() <= other._value()
