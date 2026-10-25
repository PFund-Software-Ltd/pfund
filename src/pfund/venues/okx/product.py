from enum import StrEnum

from pfund_blocks.entities.products.product_base import BaseProduct


class OKXProduct(BaseProduct):
    class Category(StrEnum):
        SWAP = "SWAP"
        FUTURES = "FUTURES"
        MARGIN = "MARGIN"
        SPOT = "SPOT"
        OPTION = "OPTION"
