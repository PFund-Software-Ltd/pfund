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
from pathlib import Path

from httpx import AsyncClient

from pfund.const.paths import CACHE_PATH
from pfund.utils.utils import load_yaml_file, dump_yaml_file, parse_raw_result
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
    SAMPLES_FILENAME = 'rest_api_samples.yml'

    URLS: ClassVar[dict[Environment, str]] = {}
    PUBLIC_ENDPOINTS: ClassVar[dict[EndpointName, tuple[RequestMethod, EndpointPath]]] = {}
    PRIVATE_ENDPOINTS: ClassVar[dict[EndpointName, tuple[RequestMethod, EndpointPath]]] = {}
    
    def __init__(self, env: Environment | tEnvironment):
        self._env = Environment[env.upper()]
        self._logger = logging.getLogger(self.name.lower())
        self._dev_mode = False
        Exchange: type[BaseExchange] = getattr(importlib.import_module(f'pfund.exchanges.{self.name.lower()}.exchange'), 'Exchange')
        self._adapter: Adapter = Exchange._adapter
        self._url: str | None = self.URLS.get(self._env, None)
        self._client = AsyncClient()
        
    def _enable_dev_mode(self):
        '''If enabled, returns only raw messages for all endpoints and stores them automatically to a yaml file as samples in caches.'''
        self._dev_mode = True
        self._logger.warning(
            "DEV mode is enabled. This mode is intended **only** for internal development of the pfund library. "
            "It should never be used in production or by end users."
        )
        
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
    
    @property
    def sample_file_path(self) -> Path:
        return CACHE_PATH / self.name / self.SAMPLES_FILENAME
    
    def _append_sample_return(self, endpoint_name: EndpointName, raw_result: RawResult):
        existing_samples = load_yaml_file(self.sample_file_path) or {}
        existing_samples[endpoint_name] = raw_result
        dump_yaml_file(self.sample_file_path, existing_samples)
    
    def get_sample_return(self, endpoint_name: EndpointName) -> RawResult:
        samples = load_yaml_file(self.sample_file_path)
        return samples[endpoint_name]
    
    async def _request(
        self,
        endpoint_name: EndpointName,
        schema: dict,
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
                raw_result: RawResult = response.raise_for_status().json()
                if self._dev_mode:
                    self._append_sample_return(endpoint_name, raw_result)
                    return raw_result
                result['request']['status_code'] = response.status_code
                is_success = response.is_success and self._is_success(raw_result)
                result['is_success'] = is_success
                result['data']['message'] = parse_raw_result(raw_result, schema) if is_success else raw_result
                if not is_success:
                    self._logger.warning(f'"{endpoint_name}" failed: {raw_result}')
            except Exception as exc:
                from httpx import RequestError, HTTPStatusError
                from json import JSONDecodeError
                result['error'] = f'{type(exc).__name__}: {exc}'
                if not isinstance(exc, (JSONDecodeError, RequestError, HTTPStatusError)):
                    self._logger.exception(f'Unhandled response exception when calling {endpoint_name}:')
        except Exception:
            self._logger.exception(f'Unhandled exception when calling {endpoint_name}:')
        return result
