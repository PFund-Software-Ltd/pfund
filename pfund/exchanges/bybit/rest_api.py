from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund.adapter import Adapter
    from pfund.exchanges.bybit.exchange import tBYBIT_PRODUCT_CATEGORY

import datetime
from pathlib import Path
import urllib
import hmac
import hashlib

from decimal import Decimal
try:
    import orjson as json
except ImportError:
    import json

from requests import Response

from pfund.exchanges.rest_api_base import BaseRestApi
from pfund.const.enums import Environment, OptionType
from pfund.exchanges.bybit.exchange import BybitProductCategory


# TODO complete the endpoints
class RestApi(BaseRestApi):
    _URLS = {
        'PAPER': 'https://api-testnet.bybit.com',
        'LIVE': 'https://api.bybit.com'
    }
    PUBLIC_ENDPOINTS = {
        # Market endpoints:
        'get_markets': ('GET', '/v5/market/instruments-info'),
    }
    PRIVATE_ENDPOINTS = {
        # Trade endpoints:
        'place_order': ('POST', '/v5/order/create'),
        'amend_order': ('POST', '/v5/order/amend'),
        'cancel_order': ('POST', '/v5/order/cancel'),
        'get_orders': ('GET', '/v5/order/realtime'),
        'cancel_all_orders': ('POST', '/v5/order/cancel-all'),
        'get_order_history': ('GET', '/v5/order/history'),
        'place_batch_orders': ('POST', '/v5/order/create-batch'),
        'amend_batch_orders': ('POST', '/v5/order/amend-batch'),
        'cancel_batch_orders': ('POST', '/v5/order/cancel-batch'),
        
        # Position endpoints:
        'get_positions': ('GET', '/v5/position/list'),
        'switch_margin_mode': ('POST', '/v5/position/switch-isolated'),
        'switch_position_mode': ('POST', '/v5/position/switch-mode'),
        'get_trades': ('GET', '/v5/execution/list'),

        # Account endpoints:
        'get_balances': ('GET', '/v5/account/wallet-balance'),
    }
    
    def __init__(self, env: Environment, adapter: Adapter):
        exch = Path(__file__).parent.name
        super().__init__(env, exch, adapter)
        
    def get_sample_return(self, func, **kwargs):
        if func == 'get_balances':
            return {'retCode': 0, 'retMsg': 'OK', 'result': {'list': [{'totalEquity': '417.77264085', 'accountIMRate': '0', 'totalMarginBalance': '417.77264085', 'totalInitialMargin': '0', 'accountType': 'UNIFIED', 'totalAvailableBalance': '417.77264085', 'accountMMRate': '0', 'totalPerpUPL': '0', 'totalWalletBalance': '417.77264085', 'accountLTV': '0', 'totalMaintenanceMargin': '0', 'coin': [{'availableToBorrow': '2500000', 'bonus': '0', 'accruedInterest': '0', 'availableToWithdraw': '417.57046575', 'totalOrderIM': '0', 'equity': '417.57046576', 'totalPositionMM': '0', 'usdValue': '417.77264085', 'unrealisedPnl': '0', 'borrowAmount': '0.0', 'totalPositionIM': '0', 'walletBalance': '417.57046576', 'cumRealisedPnl': '0', 'coin': 'USDT'}]}]}, 'retExtInfo': {}, 'time': 1681927535768}
        elif func == 'get_positions':
            return {'retCode': 0, 'retMsg': 'OK', 'result': {'nextPageCursor': '', 'category': 'linear', 'list': [{'symbol': 'BTCUSDT', 'leverage': '10', 'updatedTime': '', 'side': 'None', 'bustPrice': '', 'activePrice': '', 'avgPrice': '', 'liqPrice': '', 'riskLimitValue': '2000000', 'takeProfit': '', 'positionValue': '', 'tpslMode': '', 'riskId': 1, 'trailingStop': '', 'unrealisedPnl': '', 'markPrice': '', 'size': '0', 'stopLoss': '', 'cumRealisedPnl': '', 'positionMM': '', 'createdTime': '', 'positionIdx': 0, 'tradeMode': 0, 'positionIM': ''}]}, 'retExtInfo': {}, 'time': 1681930499896}
        elif func == 'get_orders':
            return {'retCode': 0, 'retMsg': 'OK', 'result': {'nextPageCursor': '37970e89-f36d-49c1-b8e1-bf6af6ff99e0%3A1685468270327%2C37970e89-f36d-49c1-b8e1-bf6af6ff99e0%3A1685468270327', 'category': 'linear', 'list': [{'symbol': 'BTCUSDT', 'orderType': 'Limit', 'orderLinkId': '', 'slLimitPrice': '0', 'orderId': '37970e89-f36d-49c1-b8e1-bf6af6ff99e0', 'cancelType': 'UNKNOWN', 'avgPrice': '0', 'stopOrderType': '', 'lastPriceOnCreated': '27673.4', 'orderStatus': 'New', 'takeProfit': '', 'cumExecValue': '0', 'tpslMode': 'UNKNOWN', 'smpType': 'None', 'triggerDirection': 0, 'blockTradeId': '', 'isLeverage': '', 'rejectReason': 'EC_NoError', 'price': '20000', 'orderIv': '', 'createdTime': '1685468270327', 'tpTriggerBy': '', 'positionIdx': 0, 'timeInForce': 'GTC', 'leavesValue': '20', 'updatedTime': '1685468270337', 'side': 'Buy', 'smpGroup': 0, 'triggerPrice': '', 'tpLimitPrice': '0', 'cumExecFee': '0', 'leavesQty': '0.001', 'slTriggerBy': '', 'closeOnTrigger': False, 'placeType': '', 'cumExecQty': '0', 'reduceOnly': False, 'qty': '0.001', 'stopLoss': '', 'smpOrderId': '', 'triggerBy': ''}]}, 'retExtInfo': {}, 'time': 1685479302813}
        elif func == 'get_trades':
            return {'retCode': 0, 'retMsg': 'OK', 'result': {'nextPageCursor': '315%3A0%2C284%3A0', 'category': 'linear', 'list': [{'symbol': 'BTCUSDT', 'orderType': 'UNKNOWN', 'underlyingPrice': '', 'orderLinkId': '', 'orderId': '1689724800-5-156671-1-2', 'stopOrderType': 'UNKNOWN', 'execTime': '1689724800000', 'feeRate': '0.0001', 'tradeIv': '', 'blockTradeId': '', 'markPrice': '29848.22', 'execPrice': '29848.22', 'markIv': '', 'orderQty': '0', 'orderPrice': '0', 'execValue': '59.69644', 'closedSize': '0', 'execType': 'Funding', 'side': 'Buy', 'indexPrice': '', 'leavesQty': '0', 'isMaker': False, 'execFee': '0.00596965', 'execId': '771dfca1-65aa-43b0-aa8b-abebb697490d', 'execQty': '0.002'}, {'symbol': 'BTCUSDT', 'orderType': 'Market', 'underlyingPrice': '', 'orderLinkId': '', 'orderId': '6c65d8fa-e6a2-4e97-994a-565174569e55', 'stopOrderType': 'UNKNOWN', 'execTime': '1689685728571', 'feeRate': '0.00055', 'tradeIv': '', 'blockTradeId': '', 'markPrice': '29952.5', 'execPrice': '29749.1', 'markIv': '', 'orderQty': '0.001', 'orderPrice': '31230.3', 'execValue': '29.7491', 'closedSize': '0', 'execType': 'Trade', 'side': 'Buy', 'indexPrice': '', 'leavesQty': '0', 'isMaker': False, 'execFee': '0.01636201', 'execId': 'dd0d7e23-f844-5adf-b6ec-2365d995329a', 'execQty': '0.001'}]}}
        elif func == 'place_order':
            return {'retCode': 0, 'retMsg': 'OK', 'result': {'orderId': 'a4461ee5-ad6c-48e0-85c2-09026419a98f', 'orderLinkId': '6582032293cfbeef744436da5ec6b37a'}, 'retExtInfo': {}, 'time': 1690019582575}
        elif func == 'cancel_order':
            return {'retCode': 0, 'retMsg': 'OK', 'result': {'orderId': '79d33fd7-2262-4d0a-b7a7-624f2c1fc3ad', 'orderLinkId': 'ce9e076afb5a69559b3be7f822cdfa45'}, 'retExtInfo': {}, 'time': 1690021242645}
        else:
            raise NotImplementedError(f'{func} is not implemented')

    def _authenticate(self, req, account):
        timestamp = str(self._get_nonce())
        recv_window = '5000'
        query_str = timestamp + account.key + recv_window
        if req.method == 'POST':
            query_str += json.dumps(req.json)
        elif req.method == 'GET':
            query_str += urllib.parse.urlencode(req.params)
        else:
            raise NotImplementedError(f'request method {req.method} is not supported')
        
        signature = hmac.new(
            account.secret.encode(encoding='utf-8'), 
            query_str.encode(encoding='utf-8'), 
            digestmod=hashlib.sha256
        ).hexdigest()
        
        headers = {
            "Content-Type": "application/json",
            "X-BAPI-API-KEY": account.key,
            "X-BAPI-SIGN": signature,
            "X-BAPI-SIGN-TYPE": "2",
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": recv_window,
        }
        req.headers = headers

    def _is_request_successful(self, resp: Response):
        msg = resp.json()
        is_successful = (
            resp.status_code == 200 and \
            'retCode' in msg and \
            msg['retCode'] == 0
        )
        return is_successful
    
    def get_markets(self, category: tBYBIT_PRODUCT_CATEGORY):
        schema = {
            'result': ['result', 'list'],
            'product': ['symbol'],
            'base_asset': ['baseCoin'],
            'quote_asset': ['quoteCoin'],
            'product_type': ['contractType'],
            'tick_size': ['priceFilter', 'tickSize', str],
            'lot_size': ['lotSizeFilter', 'qtyStep', str],
            'expiration': (
                'deliveryTime', 
                lambda expiration: datetime.datetime.fromtimestamp(int(expiration) / 1000, tz=datetime.timezone.utc),
                lambda expiration: expiration.strftime('%Y-%m-%d')
            ),
        }
        category = BybitProductCategory[category.upper()]
        if category == BybitProductCategory.SPOT:
            schema['product_type'] = 'SPOT'
            schema['lot_size'] = ['lotSizeFilter', 'basePrecision']
        elif category == BybitProductCategory.OPTION:
            schema['product_type'] = 'OPT'
            schema['option_type'] = (
                'optionsType', 
                lambda option_type: OptionType[option_type.upper()].value,
                str
            )
            schema['strike_price'] = (
                'symbol', 
                lambda symbol: symbol.split('-')[2], 
                Decimal, 
                str
            )
        category = category.value.lower()
        params = {'category': category}
        return super().get_markets(category, schema, params=params)