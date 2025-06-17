from enum import StrEnum


class TimeInForce(StrEnum):
    GoodTilCancelled = GTC = 'GoodTilCancelled'
    FillOrKill = FOK = 'FillOrKill'
    ImmediateOrCancel = IOC = 'ImmediateOrCancel'
    PostOnly = PO = 'PostOnly'