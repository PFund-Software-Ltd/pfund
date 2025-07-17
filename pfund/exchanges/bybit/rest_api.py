from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund.exchanges.bybit.exchange import tProductCategory
    from pfund.accounts.account_crypto import CryptoAccount
    from httpx import Request
    from pfund.exchanges.rest_api_base import Result, ApiResponse

import hmac
import inspect
import hashlib
import urllib
import datetime
from decimal import Decimal

import orjson as json

from pfund.enums import Environment, CryptoExchange, CryptoAssetType, OptionType
from pfund.exchanges.rest_api_base import BaseRestApi, RequestMethod
from pfund.exchanges.bybit.exchange import ProductCategory


# TODO complete the endpoints
class RestApi(BaseRestApi):
    exch = CryptoExchange.BYBIT

    VERSION = 'v5'
    URLS = {
        Environment.PAPER: 'https://api-testnet.bybit.com',
        Environment.LIVE: 'https://api.bybit.com',
    }
    PUBLIC_ENDPOINTS = {
        # Market endpoints:
        'get_markets': (RequestMethod.GET, f'/{VERSION}/market/instruments-info'),
    }
    PRIVATE_ENDPOINTS = {
        # Trade endpoints:
        'place_order': (RequestMethod.POST, f'/{VERSION}/order/create'),
        'amend_order': (RequestMethod.POST, f'/{VERSION}/order/amend'),
        'cancel_order': (RequestMethod.POST, f'/{VERSION}/order/cancel'),
        'get_orders': (RequestMethod.GET, f'/{VERSION}/order/realtime'),
        'cancel_all_orders': (RequestMethod.POST, f'/{VERSION}/order/cancel-all'),
        'get_order_history': (RequestMethod.GET, f'/{VERSION}/order/history'),
        'place_batch_orders': (RequestMethod.POST, f'/{VERSION}/order/create-batch'),
        'amend_batch_orders': (RequestMethod.POST, f'/{VERSION}/order/amend-batch'),
        'cancel_batch_orders': (RequestMethod.POST, f'/{VERSION}/order/cancel-batch'),
        
        # Position endpoints:
        'get_positions': (RequestMethod.GET, f'/{VERSION}/position/list'),
        'switch_margin_mode': (RequestMethod.POST, f'/{VERSION}/position/switch-isolated'),
        'switch_position_mode': (RequestMethod.POST, f'/{VERSION}/position/switch-mode'),
        'get_trades': (RequestMethod.GET, f'/{VERSION}/execution/list'),

        # Account endpoints:
        'get_balances': (RequestMethod.GET, f'/{VERSION}/account/wallet-balance'),
    }
    
    def _build_request(
        self, 
        method: RequestMethod, 
        endpoint: str, 
        account: CryptoAccount | None=None,
        params: dict | None=None, 
    ) -> Request:
        headers: dict | None = self._authenticate(account, method, params=params) if account else None
        if method == RequestMethod.POST:
            request: Request = self._client.build_request(
                method=method,
                url=endpoint,
                json=params,
                headers=headers,
            )
        elif method == RequestMethod.GET:
            request: Request = self._client.build_request(
                method=method,
                url=endpoint,
                params=params,
                headers=headers,
            )
        else:
            raise NotImplementedError(f'request method {method} is not supported')
        return request

    def _authenticate(self, account: CryptoAccount, method: RequestMethod, params: dict | None=None) -> dict:
        timestamp = str(self.nonce)
        recv_window = '5000'
        query_str = timestamp + account._key + recv_window
        if method == RequestMethod.POST:
            query_str += json.dumps(params)
        elif method == RequestMethod.GET:
            query_str += urllib.parse.urlencode(params)
        signature = hmac.new(
            account._secret.encode(encoding='utf-8'), 
            query_str.encode(encoding='utf-8'), 
            digestmod=hashlib.sha256
        ).hexdigest()
        
        headers = {
            "Content-Type": "application/json",
            "X-BAPI-API-KEY": account._key,
            "X-BAPI-SIGN": signature,
            "X-BAPI-SIGN-TYPE": "2",
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": recv_window,
        }
        return headers

    def _is_success(self, msg: dict) -> bool:
        '''Checks if the returned message means successful based on the exchange's standard'''
        return 'retCode' in msg and msg['retCode'] == 0
    
    async def get_markets(self, category: ProductCategory | tProductCategory) -> Result | ApiResponse:
        endpoint_name = inspect.currentframe().f_code.co_name
        category = ProductCategory[category.upper()]
        params = {'category': category.lower()}
        schema = {
            '@result': ['result', 'list'],
            'symbol': ['symbol'],
            'base_asset': [
                'baseCoin',
                lambda base_asset: self._adapter(base_asset, group='asset'),
            ],
            'quote_asset': [
                'quoteCoin',
                lambda quote_asset: self._adapter(quote_asset, group='asset'),
            ],
            'asset_type': [
                'contractType',
                lambda asset_type: self._adapter(asset_type, group='asset_type'),
                # NOTE: INVERSE-PERPETUAL cannot be converted to CryptoAssetType
                lambda asset_type: CryptoAssetType[asset_type.upper()] if asset_type in CryptoAssetType.__members__ else asset_type,
            ],
            'tick_size': ['priceFilter', 'tickSize', str],
            'lot_size': ['lotSizeFilter', 'qtyStep', str],
            'expiration': (
                'deliveryTime', 
                lambda expiration: None if expiration == '0' else datetime.datetime.fromtimestamp(int(expiration) / 1000, tz=datetime.timezone.utc),
            ),
            'category': category,
        }
        if schema:
            if category == ProductCategory.SPOT:
                schema['expiration'] = None
                schema['asset_type'] = CryptoAssetType.CRYPTO
                schema['lot_size'] = ['lotSizeFilter', 'basePrecision', str]
            elif category == ProductCategory.OPTION:
                schema['asset_type'] = CryptoAssetType.OPT
                schema['option_type'] = ('optionsType', lambda option_type: OptionType[option_type.upper()])
                schema['strike_price'] = ('symbol', lambda symbol: Decimal(symbol.split('-')[2]))
        result: Result | ApiResponse = await self._request(endpoint_name, schema, params=params)
        return result