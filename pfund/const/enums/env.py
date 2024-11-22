from enum import StrEnum


class Environment(StrEnum):
    BACKTEST = 'BACKTEST'
    TRAIN = 'TRAIN'
    SANDBOX = 'SANDBOX'
    PAPER = 'PAPER'
    LIVE = 'LIVE'