from __future__ import annotations

import datetime
from decimal import Decimal
from pathlib import Path

from typing import Callable, Any, TYPE_CHECKING
if TYPE_CHECKING:
    from pfund.types.bybit import Category
    from websocket import WebSocket

from pfund.exchanges.exchange_base import BaseExchange
from pfund.accounts import CryptoAccount
from pfund.products import CryptoProduct
from pfund.orders import CryptoOrder


class Exchange(BaseExchange):
    SUPPORTED_CATEGORIES = ['linear', 'inverse', 'spot', 'option']
    # NOTE, bybit only supports place_batch_orders for category `options`
    # TODO, come back to this if bybit supports more
    # SUPPORT_PLACE_BATCH_ORDERS = True
    # SUPPORT_CANCEL_BATCH_ORDERS = True
    
    PTYPE_TO_CATEGORY = {
        'PERP': 'linear',
        'FUT': 'linear',
        'IPERP': 'inverse',
        'IFUT': 'inverse',
        'SPOT': 'spot',
        'OPT': 'option',
    }

    _MAX_NUM_OF_PLACE_BATCH_ORDERS = 20
    _MAX_NUM_OF_CANEL_BATCH_ORDERS = 20

    def __init__(self, env: str):
        exch = Path(__file__).parent.name
        super().__init__(env, exch)
    
    def _create_pdt_matchings_config(
            # general to exchanges
            self, 
            markets,
            # specific to bybit
            category: Category
        ):
        schema = {
            'pdt': 'symbol',
            'bccy': 'baseCoin',
            'qccy': 'quoteCoin',
            'ptype': None,
        }
        category = category.lower()

        if category in ['linear', 'inverse']:
            schema['ptype'] = 'contractType'
        elif category == 'spot':
            # assign internal ptype directly since ptype is not provided in bybit's msg
            schema['ptype'] = 'SPOT'
        elif category == 'option':
            # assign internal ptype directly since ptype is not provided in bybit's msg
            schema['ptype'] = 'OPT'
        return super()._create_pdt_matchings_config(schema, markets, category=category)

    def _create_tick_sizes_and_lot_sizes_config(
            # general to exchanges
            self, 
            markets, 
            # specific to bybit
            category: Category
        ):
        schema = {
            'pdt': 'symbol',
            'tick_size': ['priceFilter', 'tickSize'], 
            'lot_size': None,
        }
        if category.lower() != 'spot':
            schema['lot_size'] = ['lotSizeFilter', 'qtyStep']
        else:
            schema['lot_size'] = ['lotSizeFilter', 'basePrecision']
        super()._create_tick_sizes_and_lot_sizes_config(schema, markets, tag=category)

    '''
    Functions using REST API
    TODO EXTEND
    '''
    def get_markets(
            # general to exchanges
            self, 
            # specific to bybit
            category: Category
        ):
        schema = {'result': ['result', 'list']}
        params = {'category': category}
        return super().get_markets(schema, params=params, is_write_samples=False)

    def get_balances(self, account: CryptoAccount, ccy: str='', **kwargs) -> dict[str, dict]:
        schema = {
            # result->list will return a useless list type containing a dict, 
            # need index '0' to get the real result
            # TODO, need to make sure it has really only one result 
            # so that using index 0 is safe
            'result': ['result', 'list', 0, 'coin'],
            'ts': 'time',
            'ts_adj': 1/10**3,  # since timestamp in bybit is in mts
            'ccy': 'coin',
            'data': {
                'wallet': ('walletBalance', str, Decimal),
                'available': ('availableToWithdraw', str, Decimal),
                'margin': ('equity', str, Decimal),
            },
        }
        params = {'accountType': account.type}
        if ccy:
            params['coin'] = self.adapter(ccy)
        if kwargs:
            params.update(kwargs)
        return super().get_balances(
            account,
            schema,
            params=params,
        )

    def get_positions(self, account: CryptoAccount, pdt: str='', category: Category='', **kwargs) -> dict | None:
        schema = {
            'result': ['result', 'list'],
            'ts': 'time',
            'ts_adj': 1/10**3,  # since timestamp in bybit is in mts
            'pdt': 'symbol',
            'side': 'side',
            'data': {
                'qty': ('size', str, Decimal, abs),
                'avg_px': ('avgPrice', str, Decimal),
                'liquidation_px': ('liqPrice', str, Decimal),
                'unrealized_pnl': ('unrealisedPnl', str, Decimal),
                'realized_pnl': ('cumRealisedPnl', str, Decimal),
            },
        }
        products = [self.get_product(pdt)] if pdt else list(self.products.values())
        positions = {'ts': 0.0, 'data': {}}
        categories = [category] if category else self.categories
        products_per_category = {category: [product for product in products if product.category == category] for category in categories}
        for category, products in products_per_category.items():
            if pdt:
                iterator = pdts = set(product.pdt for product in products)
            else:
                iterator = qccys = set(product.qccy for product in products)
            for element in iterator:
                params = {'category': category}
                if pdt:
                    epdt = self.adapter(element, ref_key=category)
                    params['symbol'] = epdt
                else:
                    eqccy = self.adapter(element)
                    params['settleCoin'] = eqccy
                if kwargs:
                    params.update(kwargs)
                categorized_positions = super().get_positions(
                    account,
                    schema,
                    params=params,
                )
                if categorized_positions:
                    if categorized_positions['ts']:
                        positions['ts'] = max(positions['ts'], categorized_positions['ts'])
                    if categorized_positions['data']:
                        positions['data'].update(categorized_positions['data'])
                else:
                    positions = categorized_positions
        return positions

    def get_orders(self, account: CryptoAccount, pdt: str='', category: Category='', **kwargs):
        schema = {
            'result': ['result', 'list'],
            'ts': 'time',
            'ts_adj': 1/10**3,  # since timestamp in bybit is in mts
            'pdt': 'symbol',
            'data': {
                'oid': ('orderLinkId', str),
                'eoid': ('orderId', str),
                'side': ('side', int),
                'px': ('price', str, Decimal),
                'qty': ('qty', str, Decimal, abs),
                'avg_px': ('avgPrice', str, Decimal),
                'filled_qty': ('cumExecQty', str, Decimal, abs),
                # FIXME (not sure) price that triggers a stop loss/take profit order
                'trigger_px': ('triggerPrice', str, Decimal),
                'o_type': ('orderType', str),
                'status': ('orderStatus', str),
                'tif': ('timeInForce', str),
                'is_reduce_only': ('reduceOnly', bool),
            },
        }
        products = [self.get_product(pdt)] if pdt else list(self.products.values())
        orders = {'ts': 0.0, 'data': {}, 'source': None}
        categories = [category] if category else self.categories
        products_per_category = {category: [product for product in products if product.category == category] for category in categories}
        for category, products in products_per_category.items():
            if pdt:
                iterator = pdts = set(product.pdt for product in products)
            else:
                iterator = qccys = set(product.qccy for product in products)
            for element in iterator:
                params = {'category': category}
                if pdt:
                    epdt = self.adapter(element, ref_key=category)
                    params['symbol'] = epdt
                else:
                    eqccy = self.adapter(element)
                    params['settleCoin'] = eqccy
                if kwargs:
                    params.update(kwargs)
                categorized_orders = super().get_orders(
                    account,
                    schema,
                    params=params,
                )
                if categorized_orders:
                    if categorized_orders['ts']:
                        orders['ts'] = max(orders['ts'], categorized_orders['ts'])
                    if categorized_orders['data']:
                        orders['data'].update(categorized_orders['data'])
                    orders['source'] = categorized_orders['source']
                else:
                    orders = categorized_orders
        return orders

    def get_trades(self, account: CryptoAccount, pdt: str='', category: Category='',
                   start_time: str|float=None, end_time: str|float=None,
                   is_funding_considered_as_trades=False, **kwargs):
        """
        Args:
            start_time: start time of trade history, 
                if datetime (string) in UTC is provided, supported format is '%Y-%m-%d %H:%M:%S'
                if timestamp (float) is provided, unit should be in seconds
            end_time: end time of trade history,
                if datetime (string) in UTC is provided, supported format is '%Y-%m-%d %H:%M:%S'
                if timestamp (float) is provided, unit should be in seconds
        """
        def _convert_to_date(time_):
            if type(time_) is float:
                date = datetime.datetime.fromtimestamp(time_, tz=datetime.timezone.utc)
            elif type(time_) is str:
                date = datetime.datetime.strptime(time_, date_format)
                date = date.replace(tzinfo=datetime.timezone.utc)
            return date
        schema = {
            'result': ['result', 'list'],
            'ts': 'time',
            'ts_adj': 1/10**3,  # since timestamp in bybit is in mts
            'pdt': 'symbol',
            'data': {
                'oid': ('orderLinkId', str),
                'eoid': ('orderId', str),
                'side': ('side', int),
                'px': ('orderPrice', str, Decimal),
                'qty': ('orderQty', str, Decimal, abs),
                'ltp': ('execPrice', str, Decimal),
                'ltq': ('execQty', str, Decimal, abs),
                'o_type': ('orderType', str),
                'trade_ts': ('execTime', float),
                # 'trade_id': ('execId', str),
                
                # specific to bybit
                'trade_type': ('execType', str),
            },
        }
        
        default_rollback_hours = 1
        date_format = '%Y-%m-%d %H:%M:%S'
        end_date = datetime.datetime.now(tz=datetime.timezone.utc) if end_time is None else _convert_to_date(end_time)
        start_date = end_date - datetime.timedelta(hours=default_rollback_hours) if start_time is None else _convert_to_date(start_time)
        end_time = int(end_date.timestamp() * 1000)  # bybit requires mts
        start_time = int(start_date.timestamp() * 1000)  # bybit requires mts

        trades = {'ts': 0.0, 'data': {}, 'source': None}
        categories = [category] if category else self.categories
        for category in categories:
            params = {'category': category, 'startTime': start_time, 'endTime': end_time}
            if pdt:
                epdt = self.adapter(pdt, ref_key=category)
                params['symbol'] = epdt
            if kwargs:
                params.update(kwargs)
            categorized_trades = super().get_trades(
                account,
                schema,
                params=params,
            )
            if categorized_trades:
                if categorized_trades['ts']:
                    trades['ts'] = max(trades['ts'], categorized_trades['ts'])
                if categorized_trades['data']:
                    trades['data'].update(categorized_trades['data'])
                trades['source'] = categorized_trades['source']
                
                # specific to bybit, remove all the 'Funding' trades
                if not is_funding_considered_as_trades:
                    for pdt in trades['data']:
                        for trade in trades['data'][pdt][:]:
                            if trade['trade_type'] != 'Trade':
                                trades['data'][pdt].remove(trade)
                            else:
                                del trade['trade_type']
                
                for pdt in trades['data']:
                    trades['data'][pdt] = self._combine_trades(trades['data'][pdt])
            else:
                trades = categorized_trades
        return trades

    def place_order(self, account: CryptoAccount, product: CryptoProduct, order: CryptoOrder):
        schema = {
            'result': 'result',
            'ts': 'time',
            'ts_adj': 1/10**3,  # since timestamp in bybit is in mts
            'data': {
                'oid': ('orderLinkId', str),
                'eoid': ('orderId', str),
            },
        }
        params = {
            'category': product.category, 
            'symbol': self.adapter(order.pdt, ref_key=product.category),
            'side': self.adapter(order.side, ref_key='sides'),
            'orderType': self.adapter(order.type),
            'qty': str(order.qty),
            'timeInForce': order.tif,
            'orderLinkId': order.oid,
        }
        if order.px:
            params['price'] = str(order.px)
        # REVIEW, maybe create a class BybitOrder to better handle this?
        if hasattr(order, 'isLeverage'):
            params['isLeverage'] = order.isLeverage
        if hasattr(order, 'orderFilter'):
            params['orderFilter'] = order.orderFilter
        if hasattr(order, 'orderLv'):
            params['orderLv'] = order.orderLv
        if hasattr(order, 'positionIdx'):
            params['positionIdx'] = int(order.positionIdx)
        if hasattr(order, 'closeOnTrigger'):
            params['closeOnTrigger'] = order.closeOnTrigger
        if hasattr(order, 'mmp'):
            params['mmp'] = order.mmp
        if hasattr(order, 'smpType'):
            params['smpType'] = order.smpType

        update = super().place_order(account, schema, params=params)
        # bybit's return has no order status, create it manually
        update['status'] = 'O---'
        return update

    def cancel_order(self, account: CryptoAccount, product: CryptoProduct, order: CryptoOrder):
        schema = {
            'result': 'result',
            'ts': 'time',
            'ts_adj': 1/10**3,  # since timestamp in bybit is in mts
            'data': {
                'oid': ('orderLinkId', str),
                'eoid': ('orderId', str),
            },
        }
        params = {
            'category': product.category, 
            'symbol': self.adapter(order.pdt, ref_key=product.category),
            'orderLinkId': order.oid,
            'orderId': order.eoid,
        }
        # REVIEW, maybe create a class BybitOrder to better handle this?
        if hasattr(order, 'orderFilter'):
            params['orderFilter'] = order.orderFilter
        update = super().cancel_order(account, schema, params=params)
        # bybit's return has no order status, create it manually
        update['status'] = 'C-C-'
        return update

    # NOTE, bybit only supports place_batch_orders for category `options`
    # TODO, come back to this if bybit supports more
    def place_batch_orders(self, account: CryptoAccount, product: CryptoProduct, orders: list[CryptoOrder]):
        assert len(orders) <= self._MAX_NUM_OF_PLACE_BATCH_ORDERS

    # NOTE, bybit only supports cancel_batch_orders for category `options`
    # TODO, come back to this if bybit supports more
    def cancel_batch_orders(self, account: CryptoAccount, product: CryptoProduct, orders: list[CryptoOrder]):
        assert len(orders) <= self._MAX_NUM_OF_CANCEL_BATCH_ORDERS


    '''
    Functions using WS API
    TODO EXTEND
    '''
    def orderbook_stream(self, pdts: list[str], callback: Callable[[WebSocket, str], Any], depth=None):
        '''Connects to orderbook stream
        Args:
            callback: It is run after the default callback _on_message() in ws, 
            and it won't receive the operational part thats used to handle connection and subscriptions
        '''
        for pdt in pdts:
            product = self.create_product(pdt)
            self.add_product(product)
            self.add_channel(
                'orderbook', 
                'public', 
                product=product,
                orderbook_depth=depth or self._ws_api.DEFAULT_ORDERBOOK_DEPTH,
            )
        self._ws_api.set_msg_callback(callback)
        self._ws_api.connect()
        
    def tradebook_stream(self, pdts: list[str], callback: Callable[[WebSocket, str], Any]):
        for pdt in pdts:
            product = self.create_product(pdt)
            self.add_product(product)
            self.add_channel(
                'tradebook', 
                'public', 
                product=product,
            )
        self._ws_api.set_msg_callback(callback)
        self._ws_api.connect()