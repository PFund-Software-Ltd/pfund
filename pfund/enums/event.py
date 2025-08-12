from enum import StrEnum


class Event(StrEnum):
    order = 'order'
    trade = 'trade'
    balance = 'balance'
    position = 'position'