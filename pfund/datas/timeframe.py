from __future__ import annotations

from enum import IntEnum


_TIMEFRAME_ALIASES: dict[str, tuple[str, ...]] = {
    'YEAR': ('y', 'yr', 'yrs', 'year', 'years'),
    'MONTH': ('mo', 'mon', 'mons', 'month', 'months'),
    'WEEK': ('w', 'wk', 'wks', 'week', 'weeks'),
    'DAY': ('d', 'day', 'days'),
    'HOUR': ('h', 'hr', 'hrs', 'hour', 'hours'),
    'MINUTE': ('m', 'min', 'mins', 'minute', 'minutes'),
    'SECOND': ('s', 'sec', 'secs', 'second', 'seconds'),
    'TICK': ('t', 'tick', 'ticks'),
    'QUOTE': ('q', 'quote', 'quotes'),
}
_ALIAS_TO_NAME: dict[str, str] = {
    alias: name
    for name, aliases in _TIMEFRAME_ALIASES.items()
    for alias in aliases
}


class Timeframe(IntEnum):
    QUOTE = q = -2
    TICK = t = -1
    SECOND = s = 1
    MINUTE = m = 60
    HOUR = h = 60 * 60
    # NOTE: These are not accurate, e.g. trading hours per day vary.
    DAY = d = 60 * 60 * 24
    WEEK = w = 60 * 60 * 24 * 7
    MONTH = mo = 60 * 60 * 24 * 30
    YEAR = y = 60 * 60 * 24 * 365

    @classmethod
    def _missing_(cls, value: object) -> Timeframe | None:
        if isinstance(value, str):
            name = _ALIAS_TO_NAME.get(value.lower())
            if name:
                return cls[name]
        return None

    @classmethod
    def pattern(cls) -> str:
        """Generate regex pattern matching any alias for all timeframe units."""
        return '|'.join(_ALIAS_TO_NAME.keys())

    def higher(self) -> Timeframe:
        """Rotate to the next higher timeframe. e.g. HOUR is higher than MINUTE."""
        members = list(Timeframe)
        current_index = members.index(self)
        next_index = current_index + 1
        if next_index < len(members):
            return members[next_index]
        return self  # Already at the highest

    def lower(self) -> Timeframe:
        """Rotate to the next lower timeframe."""
        members = list(Timeframe)
        current_index = members.index(self)
        prev_index = current_index - 1
        if prev_index >= 0:
            return members[prev_index]
        return self  # Already at the lowest

    def is_quote(self):
        return self == Timeframe.QUOTE

    def is_tick(self):
        return self == Timeframe.TICK

    def is_second(self):
        return self == Timeframe.SECOND

    def is_minute(self):
        return self == Timeframe.MINUTE

    def is_hour(self):
        return self == Timeframe.HOUR

    def is_day(self):
        return self == Timeframe.DAY

    def is_week(self):
        return self == Timeframe.WEEK

    def is_month(self):
        return self == Timeframe.MONTH

    def is_year(self):
        return self == Timeframe.YEAR

    @property
    def canonical(self) -> str:
        """Return the canonical short string for this timeframe. e.g. 'm' for MINUTE, 'mo' for MONTH."""
        return self.name.lower()[0] if self.name != 'MONTH' else 'mo'

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.canonical
