from enum import StrEnum


class PublicDataChannel(StrEnum):
    orderbook = quote = 'orderbook'
    tradebook = tick = 'tradebook'
    candlestick = kline = 'candlestick'


class PrivateDataChannel(StrEnum):
    balance = 'balance'
    position = 'position'
    order = 'order'
    trade = 'trade'


class DataChannelType(StrEnum):
    public = 'public'
    private = 'private'


class PFundDataChannel(StrEnum):
    logging = 'logging'
    internal = 'internal'
    # ping = 'ping'
    # pong = 'pong'
