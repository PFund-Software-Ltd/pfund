'''
A high-level REST API class for Binance, it uses RestApiSpot and RestApiDerivative
to handle different endpoints for spot and derivative trading
'''
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund.products.product_base import BaseProduct

from pathlib import Path

from pfund.exchanges.rest_api_base import BaseRestApi
from pfund.enums import Environment, CeFiProductType
from pfund.exchanges.binance.rest_api_spot import RestApiSpot
from pfund.exchanges.binance.rest_api_linear import RestApiLinear
from pfund.exchanges.binance.rest_api_inverse import RestApiInverse
from pfund.exchanges.binance.rest_api_option import RestApiOption


# TODO: "portfolio margin" is not supported yet, can only do it when its included in the testnet
# so we might need to create a new class RestApiPortfolio for it
class RestApi(BaseRestApi):
    URLS = {}
    PUBLIC_ENDPOINTS = {}
    PRIVATE_ENDPOINTS = {}
    
    def __init__(self, env: Environment):
        exch = Path(__file__).parent.name
        super().__init__(env, exch)
        self._rest_api_spot = RestApiSpot(env)
        self._rest_api_linear = RestApiLinear(env)
        self._rest_api_inverse = RestApiInverse(env)
        self._rest_api_option = RestApiOption(env)
        
    def _map_product_to_rest_api(self, product: BaseProduct):
        return {
            CeFiProductType.PERP: self._rest_api_linear,
            CeFiProductType.FUT: self._rest_api_linear,
            CeFiProductType.IPERP: self._rest_api_inverse,
            CeFiProductType.IFUT: self._rest_api_inverse,
            CeFiProductType.SPOT: self._rest_api_spot,
            CeFiProductType.OPT: self._rest_api_option,
        }[product.type]

    