from enum import StrEnum


class DataChannel(StrEnum):
    orderbook = quote = "orderbook"
    tradebook = tick = "tradebook"
    candlestick = kline = "candlestick"
    balance = "balance"
    position = "position"
    order = "order"
    trade = "trade"
