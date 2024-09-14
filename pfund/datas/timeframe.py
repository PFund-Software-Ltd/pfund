from enum import IntEnum


class TimeframeUnits(IntEnum):
    QUOTE = q = -2
    TICK = t = -1
    SECOND = s = 1
    MINUTE = m = 60
    HOUR = h = 60 * 60
    DAY = d = 60 * 60 * 24
    WEEK = w = 60 * 60 * 24 * 7  # REVIEW: 7 is not accurate
    MONTH = M = 60 * 60 * 24 * 30  # REVIEW: 30 is not accurate
    YEAR = y = 60 * 60 * 24 * 365  # REVIEW: 365 is not accurate

class Timeframe:
    def __init__(self, timeframe):
        self._timeframe = timeframe
        self.unit = TimeframeUnits[timeframe]
    
    def is_quote(self):
        return self.unit == -2

    def is_tick(self):
        return self.unit == -1

    def is_second(self):
        return self.unit == 1

    def is_minute(self):
        return self.unit == 60

    def is_hour(self):
        return self.unit == 3600

    def is_day(self):
        return self.unit == 86400

    def is_week(self):
        return self._timeframe == 'w'

    def is_month(self):
        return self._timeframe == 'M'
    
    def is_year(self):
        return self._timeframe == 'y'

    def __str__(self):
        return str(self.unit.name)

    def __repr__(self):
        return self._timeframe

    def __hash__(self):
        return self.unit.value

    def __eq__(self, other):
        if not isinstance(other, Timeframe):
            return NotImplemented
        return self.unit == other.unit

    def __ne__(self, other):
        if not isinstance(other, Timeframe):
            return NotImplemented
        return not self == other

    def __lt__(self, other):
        return self.unit < other.unit

    def __le__(self, other):
        return self.unit <= other.unit

    def __gt__(self, other):
        return self.unit > other.unit

    def __ge__(self, other):
        return self.unit >= other.unit
