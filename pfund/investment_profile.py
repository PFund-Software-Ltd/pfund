from __future__ import annotations

from dataclasses import dataclass

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund.types.literals import tTRADFI_PRODUCT_TYPE, tCEFI_PRODUCT_TYPE


# TODO
@dataclass
class InvestmentProfile:
    investment_objectives: list[str]
    risk_tolerance: str
    investment_horizon: str
    asset_classes: list[tTRADFI_PRODUCT_TYPE | tCEFI_PRODUCT_TYPE]
    diversification: str
    rebalancing_period: str