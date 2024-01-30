from enum import Enum


class TimeInForce(Enum):
    GTC = 'GoodTilCancelled'
    FOK = 'FillOrKill'
    IOC = 'ImmediateOrCancel'
    PO = 'PostOnly'