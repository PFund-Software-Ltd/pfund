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
