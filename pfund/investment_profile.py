from __future__ import annotations

from dataclasses import dataclass

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund.types.common_literals import tSUPPORTED_TRADFI_PRODUCT_TYPES, tSUPPORTED_CRYPTO_PRODUCT_TYPES


# TODO
@dataclass
class InvestmentProfile:
    investment_objectives: list[str]
    risk_tolerance: str
    investment_horizon: str
    asset_classes: list[tSUPPORTED_TRADFI_PRODUCT_TYPES | tSUPPORTED_CRYPTO_PRODUCT_TYPES]
    diversification: str
    rebalancing_period: str