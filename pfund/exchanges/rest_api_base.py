from __future__ import annotations
from typing import TYPE_CHECKING, Literal
if TYPE_CHECKING:
    from pfund.accounts import CryptoAccount
    from pfund.const.enums import Environment
    from pfund.typing.literals import tCRYPTO_EXCHANGE
    from pfund.adapter import Adapter

import time
import logging
import traceback
from pprint import pprint, pformat

from requests import Session, Request, Response

from pfund.const.enums import CeFiProductType
from pfund.utils.utils import parse_api_response_with_schema, convert_to_uppercases
from pfund.products.product_crypto import get_CryptoProduct


tENDPOINT_TYPE = Literal['public', 'private']


class BaseRestApi:
    _URLS = {}
    PUBLIC_ENDPOINTS = {}
    PRIVATE_ENDPOINTS = {}
    
    def __init__(self, env: Environment, exch: tCRYPTO_EXCHANGE, adapter: Adapter):
        self.env = env
        self.name = self.exch = exch.upper()
        self.logger = logging.getLogger(self.exch.lower())
        self._adapter = adapter
        self._url = self._URLS.get(self.env.value, '')
        self._session = Session()
        # categories used for grouping products, different exchanges have different categories
        # e.g. bybit has ['linear', 'inverse', 'spot', 'option'], okx has ['swap', 'futures', 'margin', 'spot', 'option']
        self._categories: list[str] = []
        
    @staticmethod
    def _get_nonce():
        return int(time.time() * 1000)
    
    def add_category(self, category: str):
        category = category.upper()
        if category not in self._categories:
            self._categories.append(category)

    def _is_request_successful(self, response: Response):
        return response.status_code == 200
    
    def _request(self, func: str, method: str, endpoint: str, account: CryptoAccount|None=None, params: dict|None=None, **kwargs):
        full_url = self._url + endpoint
        method = method.upper()
        if method == 'POST':
            request = Request(
                method=method, 
                url=full_url, 
                json=params, 
                **kwargs
            )
        elif method == 'GET':
            request = Request(
                method=method, 
                url=full_url, 
                params=params, 
                **kwargs
            )
        else:
            raise NotImplementedError(f'request method {method} is not supported')

        if account:
            self._authenticate(request, account)
        
        prepared_request = request.prepare()
        try:
            response = self._session.send(prepared_request)
            try:
                msg = response.json()
            except:
                msg = {'message': response.text}
            if self._is_request_successful(response):
                msg['is_success'] = True
                return msg
            else:
                error = {'is_exception': False, 'error_from': f'{self.exch.lower()}/rest_api/{func}', 'message': msg,
                         'data': {'account': account, 'endpoint': endpoint, 'params': params, 'kwargs': kwargs}}
        except:
            error = {'is_exception': True, 'error_from': f'{self.exch.lower()}/rest_api/{func}', 'message': traceback.format_exc(),
                     'data': {'account': account, 'endpoint': endpoint, 'params': params, 'kwargs': kwargs}}
        error['is_success'] = False
        self.logger.error(pformat(error, sort_dicts=False))
        # REVIEW: error is not expected to be handled, so return None
        # return error
        return None

    def _call_api(
        self, 
        func: str, 
        endpoint_type: tENDPOINT_TYPE, 
        account: CryptoAccount | None=None, 
        params: dict | None=None, 
        **kwargs
    ) -> dict | None:
        method, endpoint = self.get_endpoint(endpoint_type, func)
        return self._request(func, method, endpoint, account=account, params=params, **kwargs)

    def list_endpoints(self, type_: tENDPOINT_TYPE):
        endpoints = self.PUBLIC_ENDPOINTS if type_ == 'public' else self.PRIVATE_ENDPOINTS
        pprint(endpoints)
        return endpoints

    def get_endpoint(self, type_: tENDPOINT_TYPE, func: str):
        return self.PUBLIC_ENDPOINTS[func] if type_ == 'public' else self.PRIVATE_ENDPOINTS[func]
    
    def get_markets(self, category: str, schema: dict, params: dict | None=None, **kwargs) -> dict | None:
        if (response := self._call_api('get_markets', 'public', params=params, **kwargs)) is None:
            return None
        result: list[dict] = parse_api_response_with_schema(response, schema)
        markets = {}
        for market in result:
            ebasset, eqasset, epdt = convert_to_uppercases(
                market['base_asset'], 
                market['quote_asset'], 
                market['product']
            )
            eptype = market['product_type']
            basset, qasset, ptype = (
                self._adapter(ebasset, group='asset'),
                self._adapter(eqasset, group='asset'),
                CeFiProductType[self._adapter(eptype, group='product_type')]
            )

            # create product specifications
            specs = {}
            if ptype in [CeFiProductType.FUT, CeFiProductType.IFUT, CeFiProductType.OPT]:
                specs['expiration'] = market['expiration']
            if ptype == CeFiProductType.OPT:
                specs['option_type'] = market['option_type']
                specs['strike_price'] = market['strike_price']
            
            # create a product to get product.name, which is the full product name
            product_basis = '_'.join([basset, qasset, ptype.value])
            CryptoProduct = get_CryptoProduct(product_basis)
            product = CryptoProduct(
                bkr='CRYPTO',
                exch=self.exch,
                base_asset=basset,
                quote_asset=qasset,
                type=ptype,
                category=category,
                **specs,
            )
            
            # EXTEND to include more fields if needed
            markets[product.name] = {
                'symbol': epdt,
                'base_asset': basset,
                'quote_asset': qasset,
                'product_type': ptype.value,
                'tick_size': market['tick_size'],
                'lot_size': market['lot_size'],
                **specs,
            }
        return markets

    def get_balances(self, account: CryptoAccount, params: dict|None=None, **kwargs):
        return self._call_api('get_balances', 'private', account=account, params=params, **kwargs)

    def get_positions(self, account: CryptoAccount, params: dict|None=None, **kwargs):
        return self._call_api('get_positions', 'private', account=account, params=params, **kwargs)
    
    def get_orders(self, account: CryptoAccount, params: dict|None=None, **kwargs):
        return self._call_api('get_orders', 'private', account=account, params=params, **kwargs)

    def get_trades(self, account: CryptoAccount, params: dict|None=None, **kwargs):
        return self._call_api('get_trades', 'private', account=account, params=params, **kwargs)

    def place_order(self, account: CryptoAccount, params: dict|None=None, **kwargs):
        return self._call_api('place_order', 'private', account=account, params=params, **kwargs)

    def cancel_order(self, account: CryptoAccount, params: dict|None=None, **kwargs):
        return self._call_api('cancel_order', 'private', account=account, params=params, **kwargs)

