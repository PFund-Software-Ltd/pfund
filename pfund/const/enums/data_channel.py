from enum import StrEnum


class PublicDataChannel(StrEnum):
    ORDERBOOK = 'ORDERBOOK'
    TRADEBOOK = 'TRADEBOOK'
    CANDLESTICK = KLINE = 'KLINE'


class PrivateDataChannel(StrEnum):
    BALANCE = 'BALANCE'
    POSITION = 'POSITION'
    ORDER = 'ORDER'
    TRADE = 'TRADE'


class DataChannelType(StrEnum):
    PUBLIC = 'PUBLIC'
    PRIVATE = 'PRIVATE'
