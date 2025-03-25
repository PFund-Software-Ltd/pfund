from __future__ import annotations
from typing import TYPE_CHECKING, Literal
if TYPE_CHECKING:
    from pfund.products.product_base import BaseProduct
    from pfund.orders.order_base import BaseOrder
    from pfund.exchanges.exchange_base import BaseExchange
    from pfund.datas.data_base import BaseData
    from pfund.datas.data_time_based import TimeBasedData
    from pfund.typing import tCRYPTO_EXCHANGE, tENVIRONMENT
    from pfund.datas.data_config import DataConfig

import inspect
import importlib
from threading import Thread

from pfund.orders.order_crypto import CryptoOrder
from pfund.positions.position_crypto import CryptoPosition
from pfund.balances.balance_crypto import CryptoBalance
from pfund.accounts.account_crypto import CryptoAccount
from pfund.utils.utils import convert_to_uppercases
from pfund.brokers.broker_base import BaseBroker
from pfund.enums import CryptoExchange, PublicDataChannel, PrivateDataChannel, DataChannelType


class CryptoBroker(BaseBroker):
    def __init__(self, env: tENVIRONMENT='SANDBOX'):
        super().__init__(env, 'CRYPTO')
        self.exchanges = {}
    
    def start(self, zmq=None):
        for exch in self._accounts:
            exchange = self.get_exchange(exch)
            exchange.start()
            for acc in self._accounts[exch]:
                balances = self.get_balances(exch, acc=acc, is_api_call=True)
                self.portfolio_manager.update_balances(exch, acc, balances)
                
                positions = self.get_positions(exch, acc=acc, is_api_call=True)
                self.portfolio_manager.update_positions(exch, acc, positions)

                orders = self.get_orders(exch, acc, is_api_call=True)
                self.order_manager.update_orders(exch, acc, orders)
        super().start(zmq=zmq)

    def stop(self):
        super().stop()
        for exchange in self.exchanges.values():
            exchange.stop()

    # TODO
    def add_custom_data(self):
        pass

    def add_data(self, exch: tCRYPTO_EXCHANGE, product: str, data_config: DataConfig, **product_specs) -> list[TimeBasedData]:
        '''
        Args:
            product: product basis, defined as {base_asset}_{quote_asset}_{product_type}, e.g. BTC_USDT_PERP
            product_specs: product specifications, e.g. expiration, strike_price etc.
        '''
        exch, product_basis = exch.upper(), product.upper()
        product = self.add_product(exch, product_basis, **product_specs)
        datas: list[TimeBasedData] = self.data_manager.add_data(product, data_config=data_config)
        datas_non_resamplee = [data for data in datas if not data.is_resamplee()]
        for data in datas_non_resamplee:
            channel: PublicDataChannel = self._create_public_data_channel(data)
            self.add_channel(exch, channel, data=data)
        return datas
    
    def add_channel(
        self, 
        exch: tCRYPTO_EXCHANGE, 
        channel: PublicDataChannel | PrivateDataChannel | str,
        channel_type: Literal['public', 'private']='',
        data: BaseData | None=None
    ):
        '''
        Args:
            exch: exchange name, e.g. 'BYBIT'
            channel: PublicDataChannel, PrivateDataChannel, or exchange-specific string
                If PublicDataChannel: ORDERBOOK, TRADEBOOK, or KLINE
                If PrivateDataChannel: BALANCE, POSITION, ORDER, or TRADE
                If string: exchange-specific channel name, e.g. 'tickers.{symbol}' for Bybit
            channel_type: only required if channel is an exchange-specific string
                'public' for public data channels
                'private' for private data channels
            data: BaseData object, required for public channels
                Contains product and resolution information needed for subscription
        '''
        exchange = self.exchanges[exch]
        if channel in PublicDataChannel:
            assert data, f'data must be provided for {channel=}'
        channel_type: DataChannelType = self._create_data_channel_type(channel, channel_type=channel_type)
        exchange.add_channel(channel, channel_type, data=data)
    
    # useful when user wants to remove a private channel, since all private channels are added by default
    def remove_channel(
        self, 
        exch: tCRYPTO_EXCHANGE, 
        channel: PublicDataChannel | PrivateDataChannel | str,
        channel_type: Literal['public', 'private']='',
        data: BaseData | None=None
    ):
        exchange = self.exchanges[exch]
        channel_type: DataChannelType = self._create_data_channel_type(channel, channel_type=channel_type)
        exchange.remove_channel(channel, channel_type, data=data)

    def remove_data(self, product: BaseProduct, resolution: str):
        self.remove_product(product)
        data: BaseData = self.data_manager.remove_data(product, resolution)
        if channel := self._create_public_data_channel(data):
            self.remove_channel(product.exch, channel, data=data)
            
    def get_account(self, exch: tCRYPTO_EXCHANGE, acc: str) -> CryptoAccount | None:
        return self._accounts[exch.upper()].get(acc.upper(), None)
    
    def add_account(self, exch: tCRYPTO_EXCHANGE='', key: str='', secret: str='', name: str='', **kwargs) -> CryptoAccount:
        assert exch, 'kwarg "exch" must be provided'
        name = name.upper()
        if not (account := self.get_account(exch, name)):
            account = CryptoAccount(self.env, exch, key=key, secret=secret, name=name, **kwargs)
            exchange = self.add_exchange(exch)
            exchange.add_account(account)
            self._accounts[exch][account.name] = account
            self.logger.debug(f'added {account=}')
        else:
            self.logger.warning(f'{account=} has already been added, please make sure the account names are not duplicated')
        return account
    
    def get_product(self, exch: tCRYPTO_EXCHANGE, product_name: str) -> BaseProduct | None:
        return self._products[exch.upper()].get(product_name.upper(), None)

    def add_product(self, exch: tCRYPTO_EXCHANGE, product_basis: str, **product_specs) -> BaseProduct:
        exch, product_basis = exch.upper(), product_basis.upper()
        exchange = self.add_exchange(exch)
        # create another product object to format a correct product name
        product = exchange.create_product(product_basis, **product_specs)
        existing_product = self.get_product(exch, product.name)
        if not existing_product:
            exchange.add_product(product)
            self._products[exch][product.name] = product
            self.logger.debug(f'added {product=}')
        else:
            product = existing_product
        return product
    
    def remove_product(self, product: BaseProduct):
        exch, pdt = product.exch, product.name
        if exch in self._products and pdt in self._products[exch]:
            del self._products[exch][pdt]
        if not self._products[exch]:
            del self._products[exch]
            self.remove_exchange(exch)

    def get_exchange(self, exch: tCRYPTO_EXCHANGE) -> BaseExchange | None:
        return self.exchanges.get(exch.upper(), None)

    def add_exchange(self, exch: tCRYPTO_EXCHANGE) -> BaseExchange:
        exch = exch.upper()
        assert exch in CryptoExchange.__members__, f'exchange {exch} is not supported'
        if not (exchange := self.get_exchange(exch)):
            Exchange = getattr(importlib.import_module(f'pfund.exchanges.{exch.lower()}.exchange'), 'Exchange')
            exchange = Exchange(self.env.value)
            self.exchanges[exch] = exchange
            self.connection_manager.add_api(exchange._ws_api)
            self.logger.debug(f'added {exch=}')
        return exchange
    
    def remove_exchange(self, exch: tCRYPTO_EXCHANGE):
        exch = exch.upper()
        if exch in self.exchanges:
            exchange = self.exchanges[exch]
            del self.exchanges[exch]
            self.connection_manager.remove_api(exchange._ws_api)
            self.logger.debug(f'removed {exch=}')
    
    def add_balance(self, exch: tCRYPTO_EXCHANGE, acc: str, ccy: str) -> CryptoBalance:
        exch, acc, ccy = convert_to_uppercases(exch, acc, ccy)
        if not (balance := self.get_balances(exch, acc=acc, ccy=ccy)):
            self.add_exchange(exch)
            account = self.get_account(exch, acc)
            balance = CryptoBalance(account, ccy)
            self.portfolio_manager.add_balance(balance)
            self.logger.debug(f'added {balance=}')
        return balance

    def add_position(self, exch: tCRYPTO_EXCHANGE, acc: str, pdt: str) -> CryptoPosition:
        exch, acc, pdt = convert_to_uppercases(exch, acc, pdt)
        if not (position := self.get_positions(exch, acc=acc, pdt=pdt)):
            account = self.get_account(exch, acc)
            product = self.add_product(exch, pdt=pdt)
            position = CryptoPosition(account, product)
            self.portfolio_manager.add_position(position)
            self.logger.debug(f'added {position=}')
        return position

    def reconcile_orders(self):
        def work():
            for exch in self._accounts:
                for acc in self._accounts[exch]:
                    self.get_orders(exch, acc, is_api_call=True)
        func = inspect.stack()[0][3]
        Thread(target=work, name=func+'_thread', daemon=True).start()

    def get_orders(self, exch: tCRYPTO_EXCHANGE, acc: str, pdt: str='', oid: str='', eoid: str='', is_api_call=False, **kwargs) -> dict | None:
        exch, acc, pdt = convert_to_uppercases(exch, acc, pdt)
        if not is_api_call:
            return self.order_manager.get_orders(exch, acc, pdt=pdt, oid=oid, eoid=eoid)
        else:
            exchange = self.get_exchange(exch)
            account = self.get_account(exch, acc)
            return exchange.get_orders(account, pdt=pdt, **kwargs)

    def reconcile_trades(self):
        def work():
            for exch in self._accounts:
                for acc in self._accounts[exch]:
                    self.get_trades(exch, acc, is_api_call=True)
        func = inspect.stack()[0][3]
        Thread(target=work, name=func+'_thread', daemon=True).start()

    def get_trades(self, exch: tCRYPTO_EXCHANGE, acc: str, pdt: str='', is_api_call=False, **kwargs) -> dict | None:
        exch, acc, pdt = convert_to_uppercases(exch, acc, pdt)
        if not is_api_call:
            return self.order_manager.get_trades(...)
        else:
            exchange = self.get_exchange(exch)
            account = self.get_account(exch, acc)
            return exchange.get_trades(account, pdt=pdt, **kwargs)

    def reconcile_balances(self):
        def work():
            for exch in self._accounts:
                for acc in self._accounts[exch]:
                    self.get_balances(exch, acc, is_api_call=True)
        func = inspect.stack()[0][3]
        Thread(target=work, name=func+'_thread', daemon=True).start()

    def get_balances(self, exch: tCRYPTO_EXCHANGE, acc: str='', ccy: str='', is_api_call=False, **kwargs) -> dict | None:
        exch, acc, ccy = convert_to_uppercases(exch, acc, ccy)
        if not is_api_call:
            return self.portfolio_manager.get_balances(exch, acc, ccy=ccy)
        else:
            exchange = self.get_exchange(exch)
            account = self.get_account(exch, acc)
            return exchange.get_balances(account, ccy=ccy, **kwargs)

    def reconcile_positions(self):
        def work():
            for exch in self._accounts:
                for acc in self._accounts[exch]:
                    self.get_positions(exch, acc, is_api_call=True)
        func = inspect.stack()[0][3]
        Thread(target=work, name=func+'_thread', daemon=True).start()

    def get_positions(self, exch: tCRYPTO_EXCHANGE, acc: str='', pdt: str='', is_api_call=False, **kwargs) -> dict | None:
        exch, acc, pdt = convert_to_uppercases(exch, acc, pdt)
        if not is_api_call:
            return self.portfolio_manager.get_positions(exch, acc=acc, pdt=pdt)
        else:
            exchange = self.get_exchange(exch)
            account = self.get_account(exch, acc)
            return exchange.get_positions(account, pdt=pdt, **kwargs)

    def create_order(self, account: CryptoAccount, product: BaseProduct, side: int, quantity: float, price=None, **kwargs):
        return CryptoOrder(account, product, side, qty=quantity, px=price, **kwargs)
    
    def place_orders(self, account: CryptoAccount, product: BaseProduct, orders: list[BaseOrder]):
        exchange = self.get_exchange(account.exch)
        # TODO
        # self.rm checks risk

        num_orders = 0
        for o in orders:
            self.order_manager.on_submitted(o)
            num_orders += 1
        
        if exchange.SUPPORT_PLACE_BATCH_ORDERS and num_orders > 1:
            place_orders = exchange.place_batch_orders
            orders = [orders]
        else:
            place_orders = exchange.place_order

        # REVIEW: performance issue if sending too many orders all at once and 
        # the exchange doesn't support batch orders
        for order_s in orders:
            if not exchange.USE_WS_PLACE_ORDER:
                Thread(target=place_orders, args=(account, product, order_s,), daemon=True).start()
            # TODO
            else:
                ws_msg = place_orders(account, product, order_s)
                # NOTE: if exchange uses ws api to place order, it will return a ws_msg for ws
                # and this msg will be sent to the start_process() in connection_manager.py
                if ws_msg is not None:
                    self._zmq.send(ws_msg)

    def cancel_orders(self, account: CryptoAccount, product: BaseProduct, orders: list[BaseOrder]):
        exchange = self.get_exchange(account.exch)

        num_orders = 0
        for o in orders:
            self.order_manager.on_cancel(o)
            num_orders += 1

        if exchange.SUPPORT_CANCEL_BATCH_ORDERS and num_orders > 1:
            cancel_orders = exchange.SUPPORT_CANCEL_BATCH_ORDERS
            orders = [orders]
        else:
            cancel_orders = exchange.cancel_order

        # REVIEW: performance issue if cancelling too many orders all at once and 
        # the exchange doesn't support batch orders
        for order_s in orders:
            if not exchange.USE_WS_CANCEL_ORDER:
                Thread(target=cancel_orders, args=(account, product, order_s,), daemon=True).start()
            # TODO
            else:
                ws_msg = cancel_orders(account, product, order_s)
                # NOTE: if exchange uses ws api to cancel order, it will return a ws_msg for ws
                # and this msg will be sent to the start_process() in connection_manager.py
                if ws_msg is not None:
                    self._zmq.send(ws_msg)
    
    # TODO
    def cancel_all_orders(self):
        pass

    # TODO
    def amend_orders(self, account: CryptoAccount, product: BaseProduct, orders: list[BaseOrder]):
        # TODO, self.rm checks risk
        # if failed risk check, reset amend_px and amend_qty

        for o in orders:
            self.order_manager.on_amend(o)
