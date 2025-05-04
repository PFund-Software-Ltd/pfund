from enum import StrEnum


class Signal(StrEnum):
    ready = 'ready'
    start = 'start'
    stop = 'stop'