from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund.typing import ResolutionRepr

import re


# TODO use total_ordering from functools
class Resolution:
    DEFAULT_ORDERBOOK_LEVEL = 1
    
    def __init__(self, resolution: Resolution | ResolutionRepr):
        '''
        Args:
            resolution: e.g. '1m', '1minute', '1quote_L1'
            If the input is a data_type (e.g. 'minute', 'daily'), 
            it will be converted to resolution by adding '1' to the beginning.
            e.g. 'minute' -> '1m', 'daily' -> '1d'
        '''
        from pfund.datas.timeframe import Timeframe

        if isinstance(resolution, Resolution):
            # Copy all attributes from the resolution object
            self.__dict__.update(resolution.__dict__)
            return

        # Add "1" if the resolution doesn't start with a number
        if not re.match(r"^\d", resolution):
            resolution = "1" + resolution
        # Only remove hyphens and underscores after the initial numbers
        resolution = re.sub(r'^(\d+)[-_]', r'\1', resolution)
        assert re.match(r"^\d+[a-zA-Z]+(?:_L[1-3])?$", resolution), f"Invalid {resolution=}, pattern should be e.g. '1d', '2m', '3h', '1quote_L1' etc."
 
        resolution, *orderbook_level = resolution.strip().split('_')
        self._resolution = self._standardize(resolution)

        # split resolution (e.g. '1m') into period (e.g. '1') and timeframe (e.g. 'm')
        period, timeframe = re.split(r'(\d+)', self._resolution.strip())[1:]
        self.period = int(period)
        assert self.period > 0
        self.timeframe = Timeframe(timeframe)
        if self.is_quote():
            if orderbook_level:
                self.orderbook_level = int(orderbook_level[0][-1])
            else:
                self.orderbook_level = self.DEFAULT_ORDERBOOK_LEVEL
                print("\033[1m" + f"Warning: {self._resolution=} is missing orderbook level, defaulting to L{self.DEFAULT_ORDERBOOK_LEVEL}" + "\033[0m")
        else:
            self.orderbook_level = None

    def _standardize(self, resolution: str) -> str:
        '''Standardize resolution
        e.g. convert '1minute' to '1m'
        '''
        period, timeframe = re.split(r'(\d+)', resolution.strip())[1:]
        if timeframe.lower() in ['mon', 'mons', 'month', 'months'] or timeframe == 'M':
            timeframe = 'M'
        else:
            timeframe = timeframe[0].lower()
        standardized_resolution = period + timeframe
        return standardized_resolution

    def to_seconds(self) -> int:
        assert self.is_bar(), f'{self._resolution=} is not a bar resolution'
        return self._value()
    
    def _value(self) -> int:
        '''lower value = higher resolution'''
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
        return (
            self.is_second() or
            self.is_minute() or
            self.is_hour() or
            self.is_day()
        )
    
    def is_second(self):
        return self.timeframe.is_second()

    def is_minute(self):
        return self.timeframe.is_minute()

    def is_hour(self):
        return self.timeframe.is_hour()

    def is_day(self):
        return self.timeframe.is_day()

    def higher(self) -> Resolution:
        '''Rotate to the next higher resolution. e.g. 1m > 1h, higher resolution = lower timeframe'''
        if self.is_quote():
            if self.orderbook_level < 3:
                return Resolution('1' + repr(self.timeframe) + '_L' + str(self.orderbook_level + 1))
            else:
                return self
        else:
            return Resolution('1' + repr(self.timeframe.lower()))
    
    def lower(self) -> Resolution:
        '''Rotate to the next lower resolution. e.g. 1h < 1m, lower resolution = higher timeframe'''
        if self.is_quote() and self.orderbook_level > 1:
            return Resolution('1' + repr(self.timeframe) + '_L' + str(self.orderbook_level - 1))
        else:
            return Resolution('1' + repr(self.timeframe.higher()))
        
    def to_unit(self) -> Resolution:
        '''Convert to unit resolution. e.g. 5m -> 1m'''
        timeframe = repr(self.timeframe)
        if self.is_quote():
            return Resolution('1' + timeframe + '_L' + str(self.orderbook_level))
        else:
            return Resolution('1' + timeframe)
    
    def get_higher_resolutions(self, ignore_period: bool=False, exclude_quote: bool=False) -> list[Resolution]:
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
    
    def get_lower_resolutions(self, exclude_quote: bool=False) -> list[Resolution]:
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
        strings = [str(self.period), str(self.timeframe)]
        if self.orderbook_level:
            strings.append(f'LEVEL{self.orderbook_level}')
        return '_'.join(strings)

    def __repr__(self):
        strings = [self._resolution]
        if self.orderbook_level:
            strings.append(f'L{self.orderbook_level}')
        return '_'.join(strings)

    def __hash__(self):
        return self._value()
    
    def is_strict_equal(self, other):
        '''1h = 60m when using __eq__ to compare resolutions, but in strict_equal, 1h != 60m'''
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
