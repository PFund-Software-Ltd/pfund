from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund.products.product_bybit import tPRODUCT_CATEGORY
    from pfund.accounts.account_crypto import CryptoAccount
    from httpx import Request, Response
    from pfund.exchanges.rest_api_base import Result, RawResult

import hmac
import inspect
import hashlib
import urllib
import datetime
from decimal import Decimal

import orjson as json

from pfund.enums import Environment, CryptoExchange, CryptoAssetType, OptionType
from pfund.exchanges.rest_api_base import BaseRestApi
from pfund.products.product_bybit import ProductCategory
from pfund.exchanges.rest_api_base import RequestMethod


# TODO complete the endpoints
class RestApi(BaseRestApi):
    name = CryptoExchange.BYBIT

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
    
    def get_sample_return(self, endpoint_name, **kwargs):
        if endpoint_name == 'get_balances':
            return {'retCode': 0, 'retMsg': 'OK', 'result': {'list': [{'totalEquity': '417.77264085', 'accountIMRate': '0', 'totalMarginBalance': '417.77264085', 'totalInitialMargin': '0', 'accountType': 'UNIFIED', 'totalAvailableBalance': '417.77264085', 'accountMMRate': '0', 'totalPerpUPL': '0', 'totalWalletBalance': '417.77264085', 'accountLTV': '0', 'totalMaintenanceMargin': '0', 'coin': [{'availableToBorrow': '2500000', 'bonus': '0', 'accruedInterest': '0', 'availableToWithdraw': '417.57046575', 'totalOrderIM': '0', 'equity': '417.57046576', 'totalPositionMM': '0', 'usdValue': '417.77264085', 'unrealisedPnl': '0', 'borrowAmount': '0.0', 'totalPositionIM': '0', 'walletBalance': '417.57046576', 'cumRealisedPnl': '0', 'coin': 'USDT'}]}]}, 'retExtInfo': {}, 'time': 1681927535768}
        elif endpoint_name == 'get_positions':
            return {'retCode': 0, 'retMsg': 'OK', 'result': {'nextPageCursor': '', 'category': 'linear', 'list': [{'symbol': 'BTCUSDT', 'leverage': '10', 'updatedTime': '', 'side': 'None', 'bustPrice': '', 'activePrice': '', 'avgPrice': '', 'liqPrice': '', 'riskLimitValue': '2000000', 'takeProfit': '', 'positionValue': '', 'tpslMode': '', 'riskId': 1, 'trailingStop': '', 'unrealisedPnl': '', 'markPrice': '', 'size': '0', 'stopLoss': '', 'cumRealisedPnl': '', 'positionMM': '', 'createdTime': '', 'positionIdx': 0, 'tradeMode': 0, 'positionIM': ''}]}, 'retExtInfo': {}, 'time': 1681930499896}
        elif endpoint_name == 'get_orders':
            return {'retCode': 0, 'retMsg': 'OK', 'result': {'nextPageCursor': '37970e89-f36d-49c1-b8e1-bf6af6ff99e0%3A1685468270327%2C37970e89-f36d-49c1-b8e1-bf6af6ff99e0%3A1685468270327', 'category': 'linear', 'list': [{'symbol': 'BTCUSDT', 'orderType': 'Limit', 'orderLinkId': '', 'slLimitPrice': '0', 'orderId': '37970e89-f36d-49c1-b8e1-bf6af6ff99e0', 'cancelType': 'UNKNOWN', 'avgPrice': '0', 'stopOrderType': '', 'lastPriceOnCreated': '27673.4', 'orderStatus': 'New', 'takeProfit': '', 'cumExecValue': '0', 'tpslMode': 'UNKNOWN', 'smpType': 'None', 'triggerDirection': 0, 'blockTradeId': '', 'isLeverage': '', 'rejectReason': 'EC_NoError', 'price': '20000', 'orderIv': '', 'createdTime': '1685468270327', 'tpTriggerBy': '', 'positionIdx': 0, 'timeInForce': 'GTC', 'leavesValue': '20', 'updatedTime': '1685468270337', 'side': 'Buy', 'smpGroup': 0, 'triggerPrice': '', 'tpLimitPrice': '0', 'cumExecFee': '0', 'leavesQty': '0.001', 'slTriggerBy': '', 'closeOnTrigger': False, 'placeType': '', 'cumExecQty': '0', 'reduceOnly': False, 'qty': '0.001', 'stopLoss': '', 'smpOrderId': '', 'triggerBy': ''}]}, 'retExtInfo': {}, 'time': 1685479302813}
        elif endpoint_name == 'get_trades':
            return {'retCode': 0, 'retMsg': 'OK', 'result': {'nextPageCursor': '315%3A0%2C284%3A0', 'category': 'linear', 'list': [{'symbol': 'BTCUSDT', 'orderType': 'UNKNOWN', 'underlyingPrice': '', 'orderLinkId': '', 'orderId': '1689724800-5-156671-1-2', 'stopOrderType': 'UNKNOWN', 'execTime': '1689724800000', 'feeRate': '0.0001', 'tradeIv': '', 'blockTradeId': '', 'markPrice': '29848.22', 'execPrice': '29848.22', 'markIv': '', 'orderQty': '0', 'orderPrice': '0', 'execValue': '59.69644', 'closedSize': '0', 'execType': 'Funding', 'side': 'Buy', 'indexPrice': '', 'leavesQty': '0', 'isMaker': False, 'execFee': '0.00596965', 'execId': '771dfca1-65aa-43b0-aa8b-abebb697490d', 'execQty': '0.002'}, {'symbol': 'BTCUSDT', 'orderType': 'Market', 'underlyingPrice': '', 'orderLinkId': '', 'orderId': '6c65d8fa-e6a2-4e97-994a-565174569e55', 'stopOrderType': 'UNKNOWN', 'execTime': '1689685728571', 'feeRate': '0.00055', 'tradeIv': '', 'blockTradeId': '', 'markPrice': '29952.5', 'execPrice': '29749.1', 'markIv': '', 'orderQty': '0.001', 'orderPrice': '31230.3', 'execValue': '29.7491', 'closedSize': '0', 'execType': 'Trade', 'side': 'Buy', 'indexPrice': '', 'leavesQty': '0', 'isMaker': False, 'execFee': '0.01636201', 'execId': 'dd0d7e23-f844-5adf-b6ec-2365d995329a', 'execQty': '0.001'}]}}
        elif endpoint_name == 'place_order':
            return {'retCode': 0, 'retMsg': 'OK', 'result': {'orderId': 'a4461ee5-ad6c-48e0-85c2-09026419a98f', 'orderLinkId': '6582032293cfbeef744436da5ec6b37a'}, 'retExtInfo': {}, 'time': 1690019582575}
        elif endpoint_name == 'cancel_order':
            return {'retCode': 0, 'retMsg': 'OK', 'result': {'orderId': '79d33fd7-2262-4d0a-b7a7-624f2c1fc3ad', 'orderLinkId': 'ce9e076afb5a69559b3be7f822cdfa45'}, 'retExtInfo': {}, 'time': 1690021242645}
        else:
            raise NotImplementedError(f'{endpoint_name} is not implemented')
    
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
    
    async def get_markets(self, category: ProductCategory | tPRODUCT_CATEGORY, raw: bool=False) -> Result | RawResult:
        '''
        Args:
            raw: if True, return the raw return message from the exchange
        '''
        func_name = inspect.currentframe().f_code.co_name
        category = ProductCategory[category.upper()]
        params = {'category': category.lower()}
        schema = None if raw else {
            'result': ['result', 'list'],
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
                lambda asset_type: CryptoAssetType[asset_type.upper()],
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
        result: Result | RawResult = await self._request(func_name, schema=schema, params=params)
        return result