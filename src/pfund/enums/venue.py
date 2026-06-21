from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pfund.venues.venue_base import BaseVenue
    from pfund.entities.accounts.account_base import BaseAccount
    from pfund.entities.orders.order_base import BaseOrder
    from pfund.entities.products.product_base import BaseProduct

import importlib
from enum import StrEnum


class TradingVenue(StrEnum):
    IBKR = "IBKR"
    BYBIT = "BYBIT"

    @property
    def venue_class(self) -> type[BaseVenue]:
        if self == TradingVenue.IBKR:
            from pfund.venues.ibkr.venue import InteractiveBrokers

            return InteractiveBrokers
        elif self == TradingVenue.BYBIT:
            from pfund.venues.bybit.venue import Bybit

            return Bybit
        else:
            raise ValueError(f"Unknown venue: {self}")

    @property
    def product_class(self) -> type[BaseProduct]:
        if self == TradingVenue.IBKR:
            class_name = f"{self}Product"
        else:
            class_name = f"{self.capitalize()}Product"
        Product = getattr(
            importlib.import_module(f"pfund.entities.products.product_{self.lower()}"),
            class_name,
        )
        return Product
