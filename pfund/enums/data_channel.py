from enum import StrEnum


class PublicDataChannel(StrEnum):
    orderbook = 'orderbook'
    tradebook = 'tradebook'
    kline = 'kline'


class PrivateDataChannel(StrEnum):
    balance = 'balance'
    position = 'position'
    order = 'order'
    trade = 'trade'


class DataChannelType(StrEnum):
    public = 'public'
    private = 'private'
