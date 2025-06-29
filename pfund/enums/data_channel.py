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
    zmq_logging = 'zmq_logging'
    internal = 'internal'
    signal = 'signal'
    # ping = 'ping'
    # pong = 'pong'
    

class PFundDataTopic(StrEnum):
    pass