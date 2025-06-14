from __future__ import annotations
from typing import TYPE_CHECKING, Literal
if TYPE_CHECKING:
    from pfund.products.product_base import BaseProduct
    from pfund.orders.order_base import BaseOrder
    from pfund.exchanges.exchange_base import BaseExchange
    from pfund.datas.data_base import BaseData
    from pfund.typing import tCryptoExchange, tEnvironment

import inspect
import importlib
from threading import Thread

from pfund.enums import Environment, Broker
from pfund.orders.order_crypto import CryptoOrder
from pfund.positions.position_crypto import CryptoPosition
from pfund.balances.balance_crypto import CryptoBalance
from pfund.accounts.account_crypto import CryptoAccount
from pfund.utils.utils import convert_to_uppercases
from pfund.enums import CryptoExchange, PublicDataChannel, PrivateDataChannel, DataChannelType
from pfund.brokers.broker_trade import TradeBroker


class CryptoBroker(TradeBroker):
    name = Broker.CRYPTO
    
    def __init__(self, env: Environment | tEnvironment=Environment.SANDBOX):
        '''
        Args:
            fetch_market_configs: 
                if True, refetch markets (e.g. tick sizes, lot sizes, listed markets) from exchanges
                even if the config files exist.
                if False, markets will be automatically refetched on a weekly basis
        '''
        super().__init__(env=env)
        self.exchanges: dict[CryptoExchange, BaseExchange] = {}
    
    def start(self, zmq=None):
        for exch in self._accounts:
            exchange = self.get_exchange(exch)
            exchange.start()
            for acc in self._accounts[exch]:
                balances = self.get_balances(exch, acc=acc, is_api_call=True)
                self._portfolio_manager.update_balances(exch, acc, balances)
                
                positions = self.get_positions(exch, acc=acc, is_api_call=True)
                self._portfolio_manager.update_positions(exch, acc, positions)

                orders = self.get_orders(exch, acc, is_api_call=True)
                self._order_manager.update_orders(exch, acc, orders)
        super().start(zmq=zmq)

    def stop(self):
        super().stop()
        for exchange in self.exchanges.values():
            exchange.stop()

    def add_channel(
        self, 
        exch: tCryptoExchange, 
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
        # TODO:
        # channel: PublicDataChannel = self._create_public_data_channel(data)
        exchange = self.exchanges[exch]
        if channel in PublicDataChannel:
            assert data, f'data must be provided for {channel=}'
        channel_type: DataChannelType = self._create_data_channel_type(channel, channel_type=channel_type)
        exchange.add_channel(channel, channel_type, data=data)
    
    # useful when user wants to remove a private channel, since all private channels are added by default
    def remove_channel(
        self, 
        exch: tCryptoExchange, 
        channel: PublicDataChannel | PrivateDataChannel | str,
        channel_type: Literal['public', 'private']='',
        data: BaseData | None=None
    ):
        exchange = self.exchanges[exch]
        channel_type: DataChannelType = self._create_data_channel_type(channel, channel_type=channel_type)
        exchange.remove_channel(channel, channel_type, data=data)
            
    def get_account(self, exch: tCryptoExchange, name: str) -> CryptoAccount | None:
        return self._accounts[CryptoExchange[exch]].get(name.upper(), None)
    
    def add_account(self, exch: tCryptoExchange, name: str='', key: str='', secret: str='') -> CryptoAccount:
        if not (account := self.get_account(exch, name)):
            exchange = self.add_exchange(exch)
            account = CryptoAccount(env=self._env, exch=exch, name=name, key=key, secret=secret)
            exchange.add_account(account)
            self._accounts[exch][account.name] = account
            self._logger.debug(f'added {account=}')
        else:
            raise ValueError(f'{account=} has already been added')
        return account
    
    def get_product(self, exch: tCryptoExchange, basis: str, **specs) -> BaseProduct | None:
        from pfund.products.product_base import BaseProduct
        exch = CryptoExchange[exch.upper()]
        product_key: str = BaseProduct._create_product_key(exch, basis, **specs)
        return self._products[exch].get(product_key, None)
    
    def add_product(self, exch: tCryptoExchange, basis: str, alias: str='', **specs) -> BaseProduct:
        exchange = self.add_exchange(exch)
        # create another product object to format a correct product name
        product = exchange.create_product(basis, alias=alias, **specs)
        existing_product = self.get_product(exch, basis, )
        if not existing_product:
            exchange.add_product(product)
            self._products[exchange.name][product.key] = product
            self._logger.debug(f'added {product}')
        else:
            product = existing_product
        return product
    
    def get_exchange(self, exch: tCryptoExchange) -> BaseExchange | None:
        return self.exchanges.get(exch.upper(), None)

    def add_exchange(self, exch: tCryptoExchange) -> BaseExchange:
        exch = CryptoExchange[exch.upper()]
        if not (exchange := self.get_exchange(exch)):
            Exchange = getattr(importlib.import_module(f'pfund.exchanges.{exch.lower()}.exchange'), 'Exchange')
            exchange = Exchange(self._env.value)
            self.exchanges[exch] = exchange
            self._logger.debug(f'added {exch}')
        return exchange
    
    def remove_exchange(self, exch: tCryptoExchange):
        exch = CryptoExchange[exch.upper()]
        if exch in self.exchanges:
            del self.exchanges[exch]
            self._logger.debug(f'removed {exch}')
    
    def add_balance(self, exch: tCryptoExchange, acc: str, ccy: str) -> CryptoBalance:
        exch, acc, ccy = convert_to_uppercases(exch, acc, ccy)
        if not (balance := self.get_balances(exch, acc=acc, ccy=ccy)):
            self.add_exchange(exch)
            account = self.get_account(exch, acc)
            balance = CryptoBalance(account, ccy)
            self._portfolio_manager.add_balance(balance)
            self._logger.debug(f'added {balance=}')
        return balance

    def add_position(self, exch: tCryptoExchange, acc: str, pdt: str) -> CryptoPosition:
        exch, acc, pdt = convert_to_uppercases(exch, acc, pdt)
        if not (position := self.get_positions(exch, acc=acc, pdt=pdt)):
            account = self.get_account(exch, acc)
            product = self.add_product(exch, pdt=pdt)
            position = CryptoPosition(account, product)
            self._portfolio_manager.add_position(position)
            self._logger.debug(f'added {position=}')
        return position

    def reconcile_orders(self):
        def work():
            for exch in self._accounts:
                for acc in self._accounts[exch]:
                    self.get_orders(exch, acc, is_api_call=True)
        func = inspect.stack()[0][3]
        Thread(target=work, name=func+'_thread', daemon=True).start()

    def get_orders(self, exch: tCryptoExchange, acc: str, pdt: str='', oid: str='', eoid: str='', is_api_call=False, **kwargs) -> dict | None:
        exch, acc, pdt = convert_to_uppercases(exch, acc, pdt)
        if not is_api_call:
            return self._order_manager.get_orders(exch, acc, pdt=pdt, oid=oid, eoid=eoid)
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

    def get_trades(self, exch: tCryptoExchange, acc: str, pdt: str='', is_api_call=False, **kwargs) -> dict | None:
        exch, acc, pdt = convert_to_uppercases(exch, acc, pdt)
        if not is_api_call:
            return self._order_manager.get_trades(...)
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

    def get_balances(self, exch: tCryptoExchange, acc: str='', ccy: str='', is_api_call=False, **kwargs) -> dict | None:
        exch, acc, ccy = convert_to_uppercases(exch, acc, ccy)
        if not is_api_call:
            return self._portfolio_manager.get_balances(exch, acc, ccy=ccy)
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

    def get_positions(self, exch: tCryptoExchange, acc: str='', pdt: str='', is_api_call=False, **kwargs) -> dict | None:
        exch, acc, pdt = convert_to_uppercases(exch, acc, pdt)
        if not is_api_call:
            return self._portfolio_manager.get_positions(exch, acc=acc, pdt=pdt)
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
            self._order_manager.on_submitted(o)
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
            self._order_manager.on_cancel(o)
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
            self._order_manager.on_amend(o)
