from enum import StrEnum


class PublicDataChannel(StrEnum):
    orderbook = quote = "orderbook"
    tradebook = tick = "tradebook"
    candlestick = kline = "candlestick"


class PrivateDataChannel(StrEnum):
    balance = "balance"
    position = "position"
    order = "order"
    trade = "trade"


class DataChannel(StrEnum):
    orderbook = quote = PublicDataChannel.orderbook
    tradebook = tick = PublicDataChannel.tradebook
    candlestick = kline = PublicDataChannel.candlestick
    balance = PrivateDataChannel.balance
    position = PrivateDataChannel.position
    order = PrivateDataChannel.order
    trade = PrivateDataChannel.trade


class DataChannelType(StrEnum):
    public = "public"
    private = "private"
