from enum import StrEnum


class Timeframe(StrEnum):
    QUOTE_L1 = l1 = 'QUOTE_L1'
    QUOTE_L2 = l2 = 'QUOTE_L2'
    QUOTE_L3 = l3 = 'QUOTE_L3'
    QUOTE = q = 'QUOTE'
    TICK = t = 'TICK'
    SECOND = s = 'SECOND'
    MINUTE = m = 'MINUTE'
    HOUR = h = 'HOUR'
    DAY = d = 'DAY'
    WEEK = w = 'WEEK'
    MONTH = M = 'MONTH'
    YEAR = y = 'YEAR'
