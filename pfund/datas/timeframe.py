from typing import Literal

from pfund.const.enums.timeframe import TimeframeUnits


class Timeframe:
    SUPPORTED_TIMEFRAMES = ['q', 't', 's', 'm', 'h', 'd', 'w', 'M', 'y']
    def __init__(self, timeframe: Literal['q', 't', 's', 'm', 'h', 'd', 'w', 'M', 'y']):
        self._timeframe = timeframe
        assert timeframe in self.SUPPORTED_TIMEFRAMES, f"Invalid timeframe: {timeframe}"
        self.unit = TimeframeUnits[timeframe]
    
    def higher(self):
        """Rotate to the next higher timeframe. e.g. HOUR is higher than MINUTE."""
        sorted_units = list(TimeframeUnits)  # Enum members in sorted order
        current_index = sorted_units.index(self.unit)
        next_index = current_index + 1
        if next_index < len(sorted_units):
            name = sorted_units[next_index].name
            name = name.lower()[0] if name != 'MONTH' else 'M'
            return Timeframe(name)
        return self  # Already at the highest unit

    def lower(self):
        """Rotate to the next lower timeframe."""
        sorted_units = list(TimeframeUnits)  # Enum members in sorted order
        current_index = sorted_units.index(self.unit)
        prev_index = current_index - 1
        if prev_index >= 0:
            name = sorted_units[prev_index].name
            name = name.lower()[0] if name != 'MONTH' else 'M'
            return Timeframe(name)
        return self  # Already at the lowest unit
    
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
