from enum import StrEnum


class Environment(StrEnum):
    BACKTEST = 'BACKTEST'
    SANDBOX = 'SANDBOX'
    PAPER = 'PAPER'
    LIVE = 'LIVE'