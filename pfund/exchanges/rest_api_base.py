from __future__ import annotations
from typing import TYPE_CHECKING, Literal, ClassVar, TypeAlias
if TYPE_CHECKING:
    from pfund.adapter import Adapter
    from pfund.accounts.account_crypto import CryptoAccount
    from httpx import Request, Response

import time
import logging
import traceback
from abc import ABC, abstractmethod
from enum import StrEnum
from pprint import pprint, pformat

from httpx import AsyncClient

from pfund.enums import Environment, CryptoExchange
from pfund.utils.utils import parse_api_response_with_schema, convert_to_uppercases
from pfund.products.product_base import ProductFactory


EndpointName: TypeAlias = str
EndpointPath: TypeAlias = str


class RequestMethod(StrEnum):
    GET = 'GET'
    POST = 'POST'
    PUT = 'PUT'
    DELETE = 'DELETE'
    PATCH = 'PATCH'


class BaseRestApi(ABC):
    name: ClassVar[CryptoExchange]

    URLS: ClassVar[dict[Environment, str]] = {}
    PUBLIC_ENDPOINTS: ClassVar[dict[EndpointName, tuple[RequestMethod, EndpointPath]]] = {}
    PRIVATE_ENDPOINTS: ClassVar[dict[EndpointName, tuple[RequestMethod, EndpointPath]]] = {}
    
    def __init__(self, env: Environment, adapter: Adapter):
        self._env = env
        self._logger = logging.getLogger(self.name.lower())
        self._adapter = adapter
        self._url = self.URLS[self._env]
        self._client = AsyncClient()
        
    @property
    def nonce():
        return int(time.time() * 1000)
    
    @abstractmethod
    def _authenticate(self, request: Request, account: CryptoAccount) -> Request:
        pass
    
    def _is_success(self, response: Response) -> bool:
        return response.is_success
    
    def list_endpoints(self, endpoint_type: Literal['public', 'private']) -> None:
        endpoints = self.PUBLIC_ENDPOINTS if endpoint_type.lower() == 'public' else self.PRIVATE_ENDPOINTS
        pprint(endpoints)
    
    def _request(
        self,
        endpoint_name: EndpointName,
        account: CryptoAccount | None=None,
        # data: dict | None=None,  # FIXME
        params: dict | None=None,
        **kwargs
    ) -> dict:
        from httpx import RequestError, HTTPStatusError
        from json import JSONDecodeError

        if (is_private_endpoint := account is not None):
            method, endpoint_path = self.PRIVATE_ENDPOINTS[endpoint_name]
        else:
            method, endpoint_path = self.PUBLIC_ENDPOINTS[endpoint_name]

        # NOTE: allows access to public endpoints in backtest/sandbox environment
        if self._env in [Environment.BACKTEST, Environment.SANDBOX] and not is_private_endpoint:
            url = self.URLS[Environment.LIVE]
            endpoint = url + endpoint_path
            self._logger.warning(f'{self.name} is accessing LIVE {endpoint_path=} in {self._env} environment when calling "{endpoint_name}"')
        else:
            endpoint = self._url + endpoint_path

        if method == RequestMethod.POST:
            request: Request = self._client.build_request(
                method=method,
                url=endpoint,
                json=params,
                **kwargs
            )
        elif method == RequestMethod.GET:
            request: Request = self._client.build_request(
                method=method,
                url=endpoint,
                params=params,
                **kwargs
            )
        else:
            raise NotImplementedError(f'request method {method} is not supported')

        if account:
            request: Request = self._authenticate(request, account)
        
        result = {
            'is_success': False,
            'message': None,
            'error': None,
            'status_code': None,
            'data': {
                'exchange': self.name,
                'account': account.name if account else None,
                'endpoint_name': endpoint_name,
                'endpoint': request.url,
                # 'data': data,  # FIXME: add data to result?
                'params': params,
                'kwargs': kwargs,
            }
        }
        try:
            response: Response = self._client.send(request)
            result['status_code'] = response.status_code
            try:
                result['message'] = response.raise_for_status().json()
                result['is_success'] = self._is_success(response)
            except Exception as exc:
                result['error'] = f'{type(exc).__name__}: {exc}'
                if not isinstance(exc, (JSONDecodeError, RequestError, HTTPStatusError)):
                    self._logger.exception(f'Unhandled response exception when calling {endpoint_name}:')
        except Exception:
            self._logger.exception(f'Unhandled exception when calling {endpoint_name}:')
        return result

    def get_markets(self, schema: dict, params: dict | None=None, **kwargs) -> dict | None:
        if (response := self._request('get_markets', params=params, **kwargs)) is None:
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
            
            # create a product to get str(product), which is the full product name
            product_basis = '_'.join([basset, qasset, ptype.value])
            Product = ProductFactory(trading_venue=self.name, basis=product_basis)
            product = Product(basis=product_basis, **specs)
        
            # EXTEND to include more fields if needed
            markets[str(product)] = {
                'symbol': epdt,
                'base_asset': basset,
                'quote_asset': qasset,
                'product_type': ptype.value,
                'tick_size': market['tick_size'],
                'lot_size': market['lot_size'],
                **specs,
            }
        return markets

    def place_order(self, account: CryptoAccount, params: dict, **kwargs) -> dict | None:
        return self._request('place_order', account=account, params=params, **kwargs)

    def amend_order(self, account: CryptoAccount, params: dict, **kwargs) -> dict | None:
        return self._request('amend_order', account=account, params=params, **kwargs)

    def cancel_order(self, account: CryptoAccount, params: dict, **kwargs) -> dict | None:
        return self._request('cancel_order', account=account, params=params, **kwargs)
    