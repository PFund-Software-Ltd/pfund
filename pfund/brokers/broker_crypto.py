from __future__ import annotations
from typing import TYPE_CHECKING, Literal
if TYPE_CHECKING:
    from pfund.products.product_crypto import CryptoProduct
    from pfund.orders.order_base import BaseOrder
    from pfund.datas.data_time_based import TimeBasedData
    from pfund.exchanges.exchange_base import BaseExchange
    from pfund._typing import tCryptoExchange, tEnvironment, FullDataChannel, AccountName, ProductName
    from pfund.enums import OrderSide, PublicDataChannel

import inspect
from threading import Thread

from pfund.enums import Environment, Broker
from pfund.orders.order_crypto import CryptoOrder
from pfund.positions.position_crypto import CryptoPosition
from pfund.balances.balance_crypto import CryptoBalance
from pfund.accounts.account_crypto import CryptoAccount
from pfund.utils.utils import convert_to_uppercases
from pfund.enums import CryptoExchange, PrivateDataChannel
from pfund.brokers.broker_base import BaseBroker


class CryptoBroker(BaseBroker):
    name = Broker.CRYPTO
    
    def __init__(self, env: Environment | tEnvironment=Environment.SANDBOX):
        super().__init__(env=env)
        self.exchanges: dict[CryptoExchange, BaseExchange] = {}
    
    def start(self):
        for exch in self._accounts:
            for acc in self._accounts[exch]:
                balances = self.get_balances(exch, acc=acc, is_api_call=True)
                self._portfolio_manager.update_balances(exch, acc, balances)
                
                positions = self.get_positions(exch, acc=acc, is_api_call=True)
                self._portfolio_manager.update_positions(exch, acc, positions)

                orders = self.get_orders(exch, acc, is_api_call=True)
                self._order_manager.update_orders(exch, acc, orders)
        super().start()

    def stop(self):
        super().stop()
        for exchange in self.exchanges.values():
            exchange.stop()
    
    def _add_default_private_channels(self):
        for exch in self.exchanges:
            for channel in PrivateDataChannel:
                self.add_private_channel(exch, channel)
    
    def add_public_channel(self, exch: tCryptoExchange, channel: PublicDataChannel | FullDataChannel, data: TimeBasedData | None=None):
        exchange = self.get_exchange(exch)
        exchange.add_public_channel(channel, data=data)
    
    def add_private_channel(self, exch: tCryptoExchange, channel: PrivateDataChannel | FullDataChannel):
        exchange = self.get_exchange(exch)
        exchange.add_private_channel(channel)
        
    def get_account(self, exch: tCryptoExchange, name: AccountName) -> CryptoAccount:
        return self._accounts[CryptoExchange[exch.upper()]][name]
    
    def add_account(self, exch: tCryptoExchange, name: AccountName='', key: str='', secret: str='') -> CryptoAccount:
        exchange = self.add_exchange(exch)
        if name not in self._accounts[exchange.name]:
            account = CryptoAccount(env=self._env, exchange=exch, name=name, key=key, secret=secret)
            exchange.add_account(account)
            self._accounts[exchange.name][account.name] = account
        else:
            raise ValueError(f'account name {name} has already been added')
        return account
    
    def get_product(self, exch: tCryptoExchange, name: ProductName) -> CryptoProduct:
        '''
        Args:
            name: product name (product.name)
        '''
        return self._products[CryptoExchange[exch.upper()]][name]
    
    def add_product(self, exch: tCryptoExchange, basis: str, name: ProductName='', symbol: str='', **specs) -> CryptoProduct:
        '''
        Args:
            name: product name (product.name)
        '''
        exchange = self.add_exchange(exch)
        # create another product object to get a correct product name
        product: CryptoProduct = exchange.create_product(basis, name=name, symbol=symbol, **specs)
        if product.name not in self._products[exchange.name]:
            exchange.add_product(product)
            self._products[exchange.name][product.name] = product
        else:
            existing_product: CryptoProduct = self.get_product(exch, product.name)
            # assert products are the same with the same name
            if existing_product == product:
                product = existing_product
            else:
                raise ValueError(f'product name {name} has already been used for {existing_product}')
        return product
    
    def get_exchange(self, exch: tCryptoExchange) -> BaseExchange:
        return self.exchanges[CryptoExchange[exch.upper()]]

    def add_exchange(self, exch: tCryptoExchange) -> BaseExchange:
        exch = CryptoExchange[exch.upper()]
        if exch not in self.exchanges:
            Exchange: type[BaseExchange] = exch.exchange_class
            exchange: BaseExchange = Exchange(env=self._env)
            self.exchanges[exch] = exchange
            self._logger.debug(f'added {exch}')
        else:
            exchange: BaseExchange = self.get_exchange(exch)
        return exchange
    
    def add_balance(self, exch: tCryptoExchange, acc: str, ccy: str) -> CryptoBalance:
        exch, acc, ccy = convert_to_uppercases(exch, acc, ccy)
        if not (balance := self.get_balances(exch, acc=acc, ccy=ccy)):
            self.add_exchange(exch)
            account = self.get_account(exch, acc)
            balance = CryptoBalance(account, ccy)
            self._portfolio_manager.add_balance(balance)
            self._logger.debug(f'added {balance}')
        return balance

    def add_position(self, exch: tCryptoExchange, acc: str, pdt: str) -> CryptoPosition:
        exch, acc, pdt = convert_to_uppercases(exch, acc, pdt)
        if not (position := self.get_positions(exch, acc=acc, pdt=pdt)):
            account = self.get_account(exch, acc)
            product = self.add_product(exch, pdt=pdt)
            position = CryptoPosition(account, product)
            self._portfolio_manager.add_position(position)
            self._logger.debug(f'added {position}')
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

    def create_order(
        self, 
        creator: str, 
        account: CryptoAccount, 
        product: CryptoProduct, 
        side: OrderSide | Literal['BUY', 'SELL'] | Literal[1, -1],
        quantity: float,
        price=None, 
        **kwargs
    ):
        return CryptoOrder(creator=creator, account=account, product=product, side=side, qty=quantity, px=price, **kwargs)
    
    def place_orders(self, account: CryptoAccount, product: CryptoProduct, orders: list[BaseOrder]):
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

    def cancel_orders(self, account: CryptoAccount, product: CryptoProduct, orders: list[BaseOrder]):
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
    def amend_orders(self, account: CryptoAccount, product: CryptoProduct, orders: list[BaseOrder]):
        # TODO, self.rm checks risk
        # if failed risk check, reset amend_px and amend_qty

        for o in orders:
            self._order_manager.on_amend(o)
