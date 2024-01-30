import re

from pfund.datas.timeframe import Timeframe, TimeframeUnits
from pfund.const.commons import SUPPORTED_TIMEFRAMES


class Resolution:
    def __init__(self, resolution: str):
        self._resolution = self._standardize(resolution)
        # split resolution (e.g. '1m') into period (e.g. '1') and timeframe (e.g. 'm')
        self.period, timeframe = re.split('(\d+)', self._resolution.strip())[1:]
        self.period = int(self.period)
        assert self.period > 0
        self.timeframe = Timeframe(timeframe)

    def _standardize(self, resolution):
        '''Standardize resolution
        e.g. convert '1minute' to '1m'
        '''
        period, timeframe = re.split('(\d+)', resolution.strip())[1:]
        if timeframe not in SUPPORTED_TIMEFRAMES:
            raise NotImplementedError(f'{timeframe=} ({resolution=}) is not supported')
        if timeframe in ['months', 'month', 'M']:
            standardized_resolution = period + 'M'
        else:
            standardized_resolution = period + timeframe[0]
        return standardized_resolution

    def _value(self):
        unit: TimeframeUnits = self.timeframe.unit
        return self.period * unit.value

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
    
    def __str__(self):
        return str(self.period) + '_' + str(self.timeframe)

    def __repr__(self):
        return self._resolution

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
