import re

from pfund.datas.timeframe import Timeframe, TimeframeUnits
from pfund.const.enums import Timeframe as TimeframeEnum


class Resolution:
    def __init__(self, resolution: str):
        '''
        Args:
            resolution: e.g. '1m', '1minute', '1quote_L1'
            If the input is a data_type (e.g. 'minute', 'daily'), 
            it will be converted to resolution by adding '1' to the beginning.
            e.g. 'minute' -> '1m', 'daily' -> '1d'
        '''
        # Add "1" if the resolution doesn't start with a number
        if not re.match(r"^\d", resolution):
            resolution = "1" + resolution
        assert re.match(r"^\d+[a-zA-Z]+(?:_[lL][1-3])?$", resolution), f"Invalid {resolution=}, pattern should be e.g. '1d', '2m', '3h', '1quote_L1' etc."
        resolution, *orderbook_level = resolution.strip().split('_')
        self._resolution = self._standardize(resolution)
        # split resolution (e.g. '1m') into period (e.g. '1') and timeframe (e.g. 'm')
        self.period, timeframe = re.split('(\d+)', self._resolution.strip())[1:]
        self.period = int(self.period)
        assert self.period > 0
        self.timeframe = Timeframe(timeframe)
        if orderbook_level:
            self.orderbook_level = orderbook_level[0].upper()
        elif self.is_quote():
            self.orderbook_level = 'L1'  # Default to L1
            print("\033[1m" + f"Warning: {resolution=} is missing orderbook level, defaulting to L1" + "\033[0m")
        else:
            self.orderbook_level = ''

    def _standardize(self, resolution: str) -> str:
        '''Standardize resolution
        e.g. convert '1minute' to '1m'
        '''
        period, timeframe = re.split('(\d+)', resolution.strip())[1:]
        if timeframe in ['months', 'month', 'M', 'monthly']:
            timeframe = 'M'
        else:
            timeframe = timeframe[0].lower()
        assert timeframe in TimeframeEnum.__members__, f'{timeframe=} ({resolution=}) is not supported'
        standardized_resolution = period + timeframe
        return standardized_resolution

    def _value(self) -> int:
        unit: TimeframeUnits = self.timeframe.unit
        if self.orderbook_level:
            level: int = int(self.orderbook_level[-1])
        else:
            level = 1
        return self.period * unit.value * level

    def is_quote(self):
        return self.timeframe.is_quote()

    def is_tick(self):
        return self.timeframe.is_tick()

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
    
    def higher(self, ignore_period: bool=False):
        '''Rotate to the next higher resolution. e.g. 1m > 1h, higher resolution = lower timeframe'''
        period = str(self.period) if not ignore_period else '1'
        return Resolution(period + repr(self.timeframe.lower()))
    
    def lower(self, ignore_period: bool=False):
        '''Rotate to the next lower resolution. e.g. 1h < 1m, lower resolution = higher timeframe'''
        period = str(self.period) if not ignore_period else '1'
        return Resolution(period + repr(self.timeframe.higher()))
    
    def __str__(self):
        strings = [str(self.period), str(self.timeframe)]
        if self.orderbook_level:
            strings.append(self.orderbook_level.replace('L', 'LEVEL'))
        return '_'.join(strings)

    def __repr__(self):
        strings = [self._resolution]
        if self.orderbook_level:
            strings.append(self.orderbook_level)
        return '_'.join(strings)

    def __hash__(self):
        return self._value()
    
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
