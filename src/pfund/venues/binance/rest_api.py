"""
A high-level REST API class for Binance, it uses RestAPISpot and RestAPIDerivative
to handle different endpoints for spot and derivative trading
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pfund.entities.products.product_base import BaseProduct

from pathlib import Path

from pfund.venues.binance._rest_apis import (
    RestAPILinear,
    RestAPIInverse,
    RestAPIOption,
    RestAPISpot,
)
from pfund.venues._apis.rest_api_base import BaseRestAPI
from pfund.enums import CeFiProductType, Environment


# TODO: "portfolio margin" is not supported yet, can only do it when its included in the testnet
# so we might need to create a new class RestAPIPortfolio for it
class BinanceRestAPI(BaseRestAPI):
    URLS = {}
    PUBLIC_ENDPOINTS = {}
    PRIVATE_ENDPOINTS = {}

    def __init__(self, env: Environment):
        exch = Path(__file__).parent.name
        super().__init__(env, exch)
        self._rest_api_spot = RestAPISpot(env)
        self._rest_api_linear = RestAPILinear(env)
        self._rest_api_inverse = RestAPIInverse(env)
        self._rest_api_option = RestAPIOption(env)

    def _map_product_to_rest_api(self, product: BaseProduct):
        return {
            CeFiProductType.PERP: self._rest_api_linear,
            CeFiProductType.FUT: self._rest_api_linear,
            CeFiProductType.IPERP: self._rest_api_inverse,
            CeFiProductType.IFUT: self._rest_api_inverse,
            CeFiProductType.SPOT: self._rest_api_spot,
            CeFiProductType.OPT: self._rest_api_option,
        }[product.type]
