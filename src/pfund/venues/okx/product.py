from enum import StrEnum

from pfund.entities import BaseProduct


class OKXProduct(BaseProduct):
    class Category(StrEnum):
        SWAP = "SWAP"
        FUTURES = "FUTURES"
        MARGIN = "MARGIN"
        SPOT = "SPOT"
        OPTION = "OPTION"
