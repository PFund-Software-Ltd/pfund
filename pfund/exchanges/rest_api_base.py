from __future__ import annotations
from typing import TYPE_CHECKING, TypedDict, Literal, ClassVar, TypeAlias
if TYPE_CHECKING:
    from pfund.adapter import Adapter
    from pfund.exchanges.exchange_base import BaseExchange
    from pfund.accounts.account_crypto import CryptoAccount
    from pfund.typing import tEnvironment
    from httpx import Request, Response

import time
import logging
import importlib
from abc import ABC, abstractmethod
from enum import StrEnum

from httpx import AsyncClient

from pfund.utils.utils import parse_message_with_schema
from pfund.enums import Environment, CryptoExchange


EndpointName: TypeAlias = str
EndpointPath: TypeAlias = str
RawResult: TypeAlias = dict | list[dict]

    
class ResultData(TypedDict):
    exchange: CryptoExchange
    account: str | None
    message: dict | None
    

class RequestData(TypedDict):
    endpoint_name: str
    endpoint: str
    status_code: int | None
    params: dict | None
    kwargs: dict | None


class Result(TypedDict):
    is_success: bool
    error: str | None
    request: RequestData
    data: ResultData


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
    
    def __init__(self, env: Environment | tEnvironment):
        self._env = Environment[env.upper()]
        self._logger = logging.getLogger(self.name.lower())
        Exchange: type[BaseExchange] = getattr(importlib.import_module(f'pfund.exchanges.{self.name.lower()}.exchange'), 'Exchange')
        self._adapter: Adapter = Exchange._adapter
        self._url: str | None = self.URLS.get(self._env, None)
        self._client = AsyncClient()
        
    @property
    def nonce():
        return int(time.time() * 1000)
    
    abstractmethod
    def _build_request(
        self, 
        method: RequestMethod, 
        endpoint: str, 
        account: CryptoAccount | None=None,
        params: dict | None=None, 
        **kwargs
    ) -> Request:
        pass
    
    @abstractmethod
    def _is_success(self, msg: dict) -> bool:
        pass
    
    def list_endpoints(self, endpoint_type: Literal['public', 'private']) -> None:
        from pprint import pprint
        endpoints = self.PUBLIC_ENDPOINTS if endpoint_type.lower() == 'public' else self.PRIVATE_ENDPOINTS
        pprint(endpoints)

    def get_endpoint(self, endpoint_name: EndpointName) -> tuple[RequestMethod, str]:
        if is_public_endpoint := endpoint_name in self.PUBLIC_ENDPOINTS:
            method, endpoint_path = self.PUBLIC_ENDPOINTS[endpoint_name]
        else:
            method, endpoint_path = self.PRIVATE_ENDPOINTS[endpoint_name]
        # NOTE: allows access to public endpoints in backtest/sandbox environment
        if self._env.is_simulated and is_public_endpoint:
            live_url = self.URLS[Environment.LIVE]
            endpoint = live_url + endpoint_path
        else:
            endpoint = self._url + endpoint_path
        return method, endpoint
    
    async def _request(
        self,
        endpoint_name: EndpointName,
        schema: dict | None=None,
        account: CryptoAccount | None=None,
        # data: dict | None=None,  # FIXME
        params: dict | None=None,
        **kwargs
    ) -> Result | RawResult:
        '''
        Args:
            schema: schema to parse the returned message, if None, return the raw message
        '''
        if self._env.is_simulated:
            assert account is None, f"Simulated environment {self._env} can only access public endpoints, account should NOT be provided"
        method, endpoint = self.get_endpoint(endpoint_name)
        request: Request = self._build_request(method=method, endpoint=endpoint, account=account, params=params, **kwargs)
        result: Result = {
            'is_success': False,
            'error': None,
            'request': {
                'endpoint_name': endpoint_name,
                'endpoint': request.url,
                'status_code': None,
                # 'data': data,  # FIXME: add data to result?
                'params': params,
                'kwargs': kwargs,
            },
            'data': {
                'exchange': self.name,
                'account': account.name if account else None,
                'message': None,
            }
        }
        try:
            response: Response = await self._client.send(request)
            try:
                result['request']['status_code'] = response.status_code
                msg = response.raise_for_status().json()
                if schema is None:
                    return msg
                result['is_success'] = response.is_success and self._is_success(msg)
                if result['is_success']:
                    result['data']['message'] = parse_message_with_schema(msg, schema)
                else:
                    result['data']['message'] = msg
            except Exception as exc:
                from httpx import RequestError, HTTPStatusError
                from json import JSONDecodeError
                result['error'] = f'{type(exc).__name__}: {exc}'
                if not isinstance(exc, (JSONDecodeError, RequestError, HTTPStatusError)):
                    self._logger.exception(f'Unhandled response exception when calling {endpoint_name}:')
        except Exception:
            self._logger.exception(f'Unhandled exception when calling {endpoint_name}:')
        return result
