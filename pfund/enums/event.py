from enum import StrEnum


class Event(StrEnum):
    bar = 'bar'
    tick = 'tick'
    quote = 'quote'
    order = 'order'
    trade = 'trade'
    balance = 'balance'
    position = 'position'