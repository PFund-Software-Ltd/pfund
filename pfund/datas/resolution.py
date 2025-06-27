from __future__ import annotations

import re


# TODO use total_ordering from functools
class Resolution:
    DEFAULT_ORDERBOOK_LEVEL = 1
    
    def __init__(self, resolution: str | Resolution):
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
        period, timeframe = re.split('(\d+)', self._resolution.strip())[1:]
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
        period, timeframe = re.split('(\d+)', resolution.strip())[1:]
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

    def higher(self, ignore_period: bool=True, orderbook_level: str='L1') -> Resolution:
        '''Rotate to the next higher resolution. e.g. 1m > 1h, higher resolution = lower timeframe'''
        period = str(self.period) if not ignore_period else '1'
        return Resolution(period + repr(self.timeframe.lower()) + '_' + orderbook_level)
    
    def lower(self, ignore_period: bool=True) -> Resolution:
        '''Rotate to the next lower resolution. e.g. 1h < 1m, lower resolution = higher timeframe'''
        period = str(self.period) if not ignore_period else '1'
        return Resolution(period + repr(self.timeframe.higher()))
    
    def get_higher_resolutions(self, ignore_period: bool=True, highest_resolution: Resolution | str | None=None, orderbook_level: str='L1', exclude_quote: bool=False) -> list[Resolution]:
        higher_resolutions: list[Resolution] = []
        resolution = self
        if isinstance(highest_resolution, str):
            highest_resolution = Resolution(highest_resolution)
        while (higher_resolution := resolution.higher(ignore_period=ignore_period, orderbook_level=orderbook_level)) != resolution:
            if highest_resolution and higher_resolution > highest_resolution:
                break
            if not (exclude_quote and higher_resolution.is_quote()):
                higher_resolutions.append(higher_resolution)
            resolution = higher_resolution
        return higher_resolutions
    
    def get_lower_resolutions(self, ignore_period: bool=True, lowest_resolution: Resolution | str | None=None) -> list[Resolution]:
        lower_resolutions: list[Resolution] = []
        resolution = self
        if isinstance(lowest_resolution, str):
            lowest_resolution = Resolution(lowest_resolution)
        while (lower_resolution := resolution.lower(ignore_period=ignore_period)) != resolution:
            if lowest_resolution and lower_resolution < lowest_resolution:
                break
            lower_resolutions.append(lower_resolution)
            resolution = lower_resolution
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
