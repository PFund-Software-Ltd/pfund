from enum import StrEnum


class TimeInForce(StrEnum):
    GoodTilCancel = GTC = "GoodTilCancel"
    FillOrKill = FOK = "FillOrKill"
    ImmediateOrCancel = IOC = "ImmediateOrCancel"
