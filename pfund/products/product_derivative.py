from typing import Any

from decimal import Decimal

from pfund.products.product_base import BaseProduct


class DerivativeProduct(BaseProduct):
    underlying: str=''
    # information that requires data fetching or config loading
    contract_size: Decimal = Decimal(1)

    # contract_unit: str | None = None  # e.g. barrels
    # notional: Decimal | None = None
    # leverage: Decimal | None = None
    # initial_margin: Decimal | None = None
    # maintenance_margin: Decimal | None = None
    # settlement_type: SettlementType

    def model_post_init(self, __context: Any):
        super().model_post_init(__context)
        self.underlying = self.base_asset
        self.contract_size = self.multiplier = self.contract_size or self.multiplier
    
    @property
    def multiplier(self) -> Decimal:
        return self.contract_size