from typing import Literal

from enum import IntEnum


class TimeframeUnits(IntEnum):
    QUOTE = q = -2
    TICK = t = -1
    SECOND = s = 1
    MINUTE = m = 60
    HOUR = h = 60 * 60
    DAY = d = 60 * 60 * 24  # NOTE: This is not accurate because trading hours per day vary.


class Timeframe:
    def __init__(self, timeframe: Literal['q', 't', 's', 'm', 'h', 'd']):
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
        return self.unit == TimeframeUnits.QUOTE

    def is_tick(self):
        return self.unit == TimeframeUnits.TICK

    def is_second(self):
        return self.unit == TimeframeUnits.SECOND

    def is_minute(self):
        return self.unit == TimeframeUnits.MINUTE

    def is_hour(self):
        return self.unit == TimeframeUnits.HOUR

    def is_day(self):
        return self.unit == TimeframeUnits.DAY
    
    def __str__(self):
        return str(self.unit.name)

    def __repr__(self):
        return self.unit.name.lower()[0] if self.unit.name != 'MONTH' else 'M'

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
