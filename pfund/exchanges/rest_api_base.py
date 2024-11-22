import time
import inspect
import traceback
from pprint import pprint

from typing import Literal

from requests import Session, Request, Response

from pfund.accounts import CryptoAccount
from pfund.const.enums import Environment

class BaseRestApi:
    def __init__(self, env: Environment, exch: str):
        self.env = env
        self.name = self.exch = exch.upper()
        self._url = self.URLS.get(self.env.value, '')
        self._session = Session()

    @staticmethod
    def _get_nonce():
        return int(time.time() * 1000)

    def _is_request_successful(self, resp: Response):
        return resp.status_code == 200
    
    def _request(self, func: str, method: str, endpoint: str, account: CryptoAccount|None=None, params: dict|None=None, **kwargs):
        full_url = self._url + endpoint
        method = method.upper()
        if method == 'POST':
            req = Request(
                method=method, 
                url=full_url, 
                json=params, 
                **kwargs
            )
        elif method == 'GET':
            req = Request(
                method=method, 
                url=full_url, 
                params=params, 
                **kwargs
            )
        else:
            raise NotImplementedError(f'request method {method} is not supported')

        if account:
            self._authenticate(req, account)
        
        prepped = req.prepare()
        try:
            resp = self._session.send(prepped)
            try:
                msg = resp.json()
            except:
                msg = resp.text
            if self._is_request_successful(resp):
                return msg
            else:
                error = {'is_exception': False, 'error_from': f'{self.exch.lower()}/rest_api/{func}', 'message': msg,
                         'data': {'account': account, 'endpoint': endpoint, 'params': params, 'kwargs': kwargs}}
        except:
            error = {'is_exception': True, 'error_from': f'{self.exch.lower()}/rest_api/{func}', 'message': traceback.format_exc(),
                     'data': {'account': account, 'endpoint': endpoint, 'params': params, 'kwargs': kwargs}}
        return error

    def list_endpoints(self, type_: Literal['public', 'private']):
        endpoints = self.PUBLIC_ENDPOINTS if type_ == 'public' else self.PRIVATE_ENDPOINTS
        pprint(endpoints)
        return endpoints

    def get_endpoint(self, type_: Literal['public', 'private'], func: str):
        return self.PUBLIC_ENDPOINTS[func] if type_ == 'public' else self.PRIVATE_ENDPOINTS[func]

    def _call_api(self, func: str, endpoint_type: Literal['public', 'private'], account: CryptoAccount|None=None, params: dict|None=None, **kwargs):
        method, endpoint = self.get_endpoint(endpoint_type, func)
        ret = self._request(func, method, endpoint, account=account, params=params, **kwargs)
        return ret
        
    def get_markets(self, params: dict|None=None, **kwargs):
        return self._call_api('get_markets', 'public', params=params, **kwargs)

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

