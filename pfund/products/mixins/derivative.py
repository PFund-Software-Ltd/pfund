from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund.products.product_base import BaseProduct

from decimal import Decimal


class DerivativeMixin:
    underlying: str=''
    # TODO: information that requires data fetching or config loading
    contract_size: Decimal = Decimal(1)

    # contract_unit: str | None = None  # e.g. barrels
    # notional: Decimal | None = None
    # leverage: Decimal | None = None
    # initial_margin: Decimal | None = None
    # maintenance_margin: Decimal | None = None
    # settlement_type: SettlementType

    def __mixin_post_init__(self: DerivativeMixin | BaseProduct):
        self.underlying = self.base_asset  # TODO: inaccurate
    
    @property
    def multiplier(self) -> Decimal:
        return self.contract_size