from enum import StrEnum


class TradFiBroker(StrEnum):
    IB = 'IB'


class Broker(StrEnum):
    IB = TradFiBroker.IB
    CRYPTO = 'CRYPTO'
    DEFI = 'DEFI'