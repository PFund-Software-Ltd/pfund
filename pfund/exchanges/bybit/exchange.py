from __future__ import annotations
from typing import TYPE_CHECKING, Literal
if TYPE_CHECKING:
    from pfund.exchanges.rest_api_base import Result, ApiResponse
    from pfund.products.product_base import BaseProduct

import asyncio
import datetime
from decimal import Decimal

from pfund.products.product_bybit import BybitProduct
from pfund.enums import CryptoExchange, CryptoAssetType, AssetTypeModifier
from pfund.exchanges.exchange_base import BaseExchange
from pfund.accounts.account_crypto import CryptoAccount
from pfund.orders.order_crypto import CryptoOrder


ProductCategory = BybitProduct.ProductCategory
tProductCategory = Literal['LINEAR', 'INVERSE', 'SPOT', 'OPTION']
    
    
class Exchange(BaseExchange):
    name = CryptoExchange.BYBIT
    # REVIEW
    SUPPORTED_ASSET_TYPES: list = [
        CryptoAssetType.FUTURE,
        CryptoAssetType.PERPETUAL,
        CryptoAssetType.OPTION,
        CryptoAssetType.CRYPTO,
        CryptoAssetType.INDEX,
        AssetTypeModifier.INVERSE + '-' + CryptoAssetType.FUTURE,
        AssetTypeModifier.INVERSE + '-' + CryptoAssetType.PERPETUAL,
    ]

    # TODO: may allow configure exchange behaviours such as use place_order over place_batch_orders for rate limit control
    # def configure(self, ...):
    #     pass

    '''
    Functions using REST API
    TODO EXTEND
    '''
    async def aget_markets(self, category: tProductCategory='') -> dict[ProductCategory, Result]:
        categories = [ProductCategory[category.upper()]] if category else [category for category in ProductCategory]
        markets: dict[ProductCategory, Result] = {}
        for category in categories:
            result: Result = await self._rest_api.get_markets(category=category)
            markets[category] = result
        return markets

    def get_markets(self, category: tProductCategory='') -> dict[ProductCategory, Result]:
        return asyncio.run(self.aget_markets(category=category))

    def get_balances(self, account: CryptoAccount, ccy: str='', **kwargs) -> dict[str, dict]:
        schema = {
            # result->list will return a useless list type containing a dict, 
            # need index '0' to get the real result
            # TODO, need to make sure it has really only one result so that using index 0 is safe
            '@result': ['result', 'list', 0, 'coin'],
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

    # FIXME: remove pdt, pass in product object
    def get_positions(self, account: CryptoAccount, pdt: str='', category: tProductCategory='', **kwargs) -> dict | None:
        schema = {
            '@result': ['result', 'list'],
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
        categories = [category] if category else self._categories
        products_per_category = {category: [product for product in products if product.category == category] for category in categories}
        for category, products in products_per_category.items():
            if pdt:
                iterator = pdts = set(str(product) for product in products)
            else:
                iterator = qccys = set(product.qccy for product in products)
            for element in iterator:
                params = {'category': category}
                if pdt:
                    epdt = self.adapter(element, group=product.type)
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

    def get_orders(self, account: CryptoAccount, pdt: str='', category: tProductCategory='', **kwargs):
        schema = {
            '@result': ['result', 'list'],
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
        categories = [category] if category else self._categories
        products_per_category = {category: [product for product in products if product.category == category] for category in categories}
        for category, products in products_per_category.items():
            if pdt:
                iterator = pdts = set(str(product) for product in products)
            else:
                iterator = qccys = set(product.qccy for product in products)
            for element in iterator:
                params = {'category': category}
                if pdt:
                    epdt = self.adapter(element, group=product.type)
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

    def get_trades(self, account: CryptoAccount, pdt: str='', category: tProductCategory='',
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
            '@result': ['result', 'list'],
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
        categories = [category] if category else self._categories
        for category in categories:
            params = {'category': category, 'startTime': start_time, 'endTime': end_time}
            if pdt:
                epdt = self.adapter(pdt, group=product.type)
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

    def place_order(self, account: CryptoAccount, product: BaseProduct, order: CryptoOrder, expires_in: int=5000):
        '''
        Args:
            expires_in: time in milliseconds, specify how long the HTTP request is valid.
        '''
        schema = {
            '@result': 'result',
            'ts': 'time',
            'ts_adj': 1/10**3,  # convert bybit's milliseconds to seconds
            'data': {
                'oid': ('orderLinkId', str),
                'eoid': ('orderId', str),
            },
        }
        params = {
            'category': product.category, 
            'symbol': self.adapter(order.pdt, group=product.category),
            'side': self.adapter(order.side, group='sides'),
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

        update = super().place_order(account, schema, params=params, expires_in=expires_in)
        # bybit's return has no order status, create it manually
        update['status'] = 'O---'
        return update

    def cancel_order(self, account: CryptoAccount, product: BaseProduct, order: CryptoOrder):
        schema = {
            '@result': 'result',
            'ts': 'time',
            'ts_adj': 1/10**3,  # since timestamp in bybit is in mts
            'data': {
                'oid': ('orderLinkId', str),
                'eoid': ('orderId', str),
            },
        }
        params = {
            'category': product.category, 
            'symbol': self.adapter(order.pdt, group=product.category),
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
    def place_batch_orders(self, account: CryptoAccount, product: BaseProduct, orders: list[CryptoOrder]):
        assert len(orders) <= self.MAX_NUM_OF_PLACE_BATCH_ORDERS

    # NOTE, bybit only supports cancel_batch_orders for category `options`
    # TODO, come back to this if bybit supports more
    def cancel_batch_orders(self, account: CryptoAccount, product: BaseProduct, orders: list[CryptoOrder]):
        assert len(orders) <= self._MAX_NUM_OF_CANCEL_BATCH_ORDERS
