from __future__ import annotations
from typing import TYPE_CHECKING, TypedDict, Literal, ClassVar, TypeAlias
if TYPE_CHECKING:
    from pfund._typing import tEnvironment
    from pfund.adapter import Adapter
    from pfund.exchanges.exchange_base import BaseExchange
    from pfund.accounts.account_crypto import CryptoAccount
    from httpx import Request, Response

import time
import logging
import importlib
from abc import ABC, abstractmethod
from enum import StrEnum
from pathlib import Path

from httpx import AsyncClient, RequestError, HTTPStatusError
from json import JSONDecodeError

from pfund.errors import ParseApiResponseError
from pfund.parser import SchemaParser
from pfund.utils.utils import load_yaml_file, dump_yaml_file
from pfund.enums import Environment, CryptoExchange


EndpointName: TypeAlias = str
EndpointPath: TypeAlias = str
ApiResponse: TypeAlias = dict | list[dict]

    
class RequestData(TypedDict):
    endpoint_name: str
    endpoint: str
    status_code: int | None
    params: dict | None


class Result(TypedDict):
    is_success: bool
    error: str | None
    request: RequestData
    exchange: CryptoExchange
    account: str | None
    data: dict | list[dict] | None


class RequestMethod(StrEnum):
    GET = 'GET'
    POST = 'POST'
    PUT = 'PUT'
    DELETE = 'DELETE'
    PATCH = 'PATCH'
    

class BaseRESTfulAPI(ABC):
    exch: ClassVar[CryptoExchange]
    
    SAMPLES_FILENAME = 'rest_api_samples.yml'

    URLS: ClassVar[dict[Environment, str]] = {}
    PUBLIC_ENDPOINTS: ClassVar[dict[EndpointName, tuple[RequestMethod, EndpointPath]]] = {}
    PRIVATE_ENDPOINTS: ClassVar[dict[EndpointName, tuple[RequestMethod, EndpointPath]]] = {}
    
    def __init__(self, env: Environment | tEnvironment):
        self._env = Environment[env.upper()]
        assert self._env != Environment.BACKTEST, f'{self._env=} is not supported in RESTful API'
        self._logger = logging.getLogger(self.exch.lower())
        self._dev_mode = False
        Exchange: type[BaseExchange] = getattr(importlib.import_module(f'pfund.exchanges.{self.exch.lower()}.exchange'), 'Exchange')
        self._adapter: Adapter = Exchange.adapter
        self._url: str | None = self.URLS.get(self._env, None)
        self._client = AsyncClient()
        
    def __getstate__(self) -> object:
        state = self.__dict__.copy()
        state['_client'] = None  # remove client to avoid pickling error
        return state
    
    def __setstate__(self, state: object):
        self.__dict__.update(state)
        self._client = AsyncClient()
    
    def _enable_dev_mode(self):
        '''If enabled, returns only raw messages for all endpoints and stores them automatically to a yaml file as samples in caches.'''
        self._dev_mode = True
        self._logger.warning(
            "DEV mode is enabled. This mode is intended **only** for internal development of the pfund library. "
            "It should never be used in production or by end users."
        )
        
    @property
    def nonce(self):
        return int(time.time() * 1000)
    
    @abstractmethod
    def _build_request(
        self, 
        method: RequestMethod, 
        endpoint: str, 
        account: CryptoAccount | None=None,
        params: dict | None=None, 
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
        # NOTE: allows access to public endpoints in sandbox environment
        if self._env == Environment.SANDBOX and is_public_endpoint:
            live_url = self.URLS[Environment.LIVE]
            endpoint = live_url + endpoint_path
            self._logger.warning(f'{self._env} environment is using LIVE data for public endpoint "{endpoint_name}"')
        else:
            endpoint = self._url + endpoint_path
        return method, endpoint
    
    @property
    def sample_file_path(self) -> Path:
        from pfund.config import get_config
        config = get_config()
        return Path(config.cache_path) / self.exch / self.SAMPLES_FILENAME
    
    def _append_sample_return(self, endpoint_name: EndpointName, api_response: ApiResponse):
        existing_samples = load_yaml_file(self.sample_file_path) or {}
        existing_samples[endpoint_name] = api_response
        dump_yaml_file(self.sample_file_path, existing_samples)
    
    def get_sample_return(self, endpoint_name: EndpointName) -> ApiResponse:
        samples = load_yaml_file(self.sample_file_path)
        return samples[endpoint_name]
    
    async def _request(
        self,
        endpoint_name: EndpointName,
        schema: dict,
        account: CryptoAccount | None=None,
        # data: dict | None=None,  # FIXME
        params: dict | None=None,
    ) -> Result | ApiResponse:
        '''
        Args:
            schema: schema to parse the returned message, if None, return the raw message
        '''
        if self._env.is_simulated():
            assert account is None, f"Simulated environment {self._env} can only access public endpoints, account should NOT be provided"

        method, endpoint = self.get_endpoint(endpoint_name)
        request: Request = self._build_request(method=method, endpoint=endpoint, account=account, params=params)
        result: Result = {
            'is_success': False,
            'error': None,
            'request': {
                'endpoint_name': endpoint_name,
                'endpoint': request.url,
                'status_code': None,
                # 'data': data,  # FIXME: add data to result?
                'params': params,
            },
            'exchange': self.exch,
            'account': account.name if account else None,
            'data': None,
        }
        try:
            response: Response = await self._client.send(request)
            api_response: ApiResponse = response.raise_for_status().json()
            if self._dev_mode:
                self._append_sample_return(endpoint_name, api_response)
                return api_response
            result['request']['status_code'] = response.status_code
            is_success = response.is_success and self._is_success(api_response)
            result['is_success'] = is_success
            parsed_data: dict | list[dict] = SchemaParser.convert(api_response, schema) if is_success else api_response
            if isinstance(parsed_data, dict):
                result.update(parsed_data)
            else:
                result['data'] = parsed_data
            if not is_success:
                self._logger.warning(f'"{endpoint_name}" failed: {api_response}')
        except (ParseApiResponseError, JSONDecodeError, RequestError, HTTPStatusError) as exc:
            result['error'] = f'{type(exc).__name__}: {exc}'
            self._logger.exception(f'REST API "{endpoint_name}" error:')
        except Exception as exc:
            result['error'] = f'{type(exc).__name__}: {exc}'
            self._logger.exception(f'Unhandled REST API exception when calling "{endpoint_name}":')
        return result
