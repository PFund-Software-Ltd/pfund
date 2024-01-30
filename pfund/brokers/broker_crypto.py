import inspect
import importlib
from threading import Thread

from pfund.products.product_base import BaseProduct
from pfund.products import CryptoProduct
from pfund.orders import BaseOrder, CryptoOrder
from pfund.positions import CryptoPosition
from pfund.balances import CryptoBalance
from pfund.accounts import CryptoAccount
from pfund.utils.utils import convert_to_uppercases
from pfund.brokers.broker_live import LiveBroker
from pfund.exchanges.exchange_base import BaseExchange
from pfund.const.commons import SUPPORTED_CRYPTO_EXCHANGES, SUPPORTED_CRYPTO_PRODUCT_TYPES


class CryptoBroker(LiveBroker):
    def __init__(self, env):
        super().__init__(env, 'CRYPTO')
        self.exchanges = {}
    
    def start(self, zmq=None):
        for exch in self.accounts:
            exchange = self.get_exchange(exch)
            exchange.start()
            for acc in self.accounts[exch]:
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

    def add_data(self, exch, base_currency, quote_currency, product_type, *args, **kwargs):
        exch, base_currency, quote_currency, product_type = convert_to_uppercases(exch, base_currency, quote_currency, product_type)
        product = self.add_product(exch, base_currency, quote_currency, product_type, *args, **kwargs)
        datas = self.data_manager.add_data(product, **kwargs)
        for data in datas:
            self.add_data_channel(exch, data, **kwargs)
        return datas
    
    def add_data_channel(self, exch, data, **kwargs):
        exchange = self.exchanges[exch]
        if data.is_time_based():
            if data.is_resamplee():
                return
            timeframe = data.timeframe
            if timeframe.is_quote():
                channel = 'orderbook'
            elif timeframe.is_tick():
                channel = 'tradebook'
            else:
                channel = 'kline'
            exchange.add_channel(channel, 'public', product=data.product, period=data.period, timeframe=repr(timeframe), **kwargs)
        else:
            raise NotImplementedError

    def remove_data(self, product: BaseProduct, resolution: str | None=None):
        if not (datas := self.data_manager.remove_data(product, resolution=resolution)):
            self.remove_product(product.exch, product.pdt)
            
    def get_account(self, exch: str, acc: str) -> CryptoAccount | None:
        return self.accounts[exch.upper()].get(acc.upper(), None)
    
    def add_account(self, exch: str, key: str='', secret: str='', acc: str='', **kwargs) -> CryptoAccount:
        acc = acc.upper()
        if not (account := self.get_account(exch, acc)):
            account = CryptoAccount(self.env, exch, key=key, secret=secret, acc=acc, **kwargs)
            exchange = self.add_exchange(exch)
            exchange.add_account(account)
            self.accounts[exch][acc] = account
            self.logger.debug(f'added {account=}')
        else:
            self.logger.warning(f'{account=} has already been added, please make sure the account names are not duplicated')
        return account
    
    def get_product(self, exch: str, pdt: str) -> CryptoProduct | None:
        return self.products[exch.upper()].get(pdt.upper(), None)

    def add_product(self, exch, bccy='', qccy='', ptype='', *args, pdt='', **kwargs) -> CryptoProduct:
        assert pdt or (bccy and qccy and ptype), f'Please provide `pdt` or (`bccy` and `qccy` and `ptype`)'
        assert pdt or ptype.upper() in SUPPORTED_CRYPTO_PRODUCT_TYPES, f'{self.bkr} product type {ptype} is not supported, {SUPPORTED_CRYPTO_PRODUCT_TYPES=}'
        exch, bccy, qccy, ptype, *args, pdt = convert_to_uppercases(exch, bccy, qccy, ptype, *args, pdt)
        if not pdt:
            pdt = CryptoProduct.create_product_name(bccy, qccy, ptype, *args, **kwargs)
        else:
            bccy, qccy, ptype, *args = CryptoProduct.parse_product_name(pdt)
        if not (product := self.get_product(exch, pdt)):
            exchange = self.add_exchange(exch)
            product = exchange.create_product(bccy, qccy, ptype, *args, **kwargs)
            exchange.add_product(product, **kwargs)
            self.products[exch][pdt] = product
            self.logger.debug(f'added {product=}')
        return product
    
    def remove_product(self, exch: str, pdt: str):
        exch, pdt = exch.upper(), pdt.upper()
        if exch in self.products and pdt in self.products[exch]:
            del self.products[exch][pdt]
        if not self.products[exch]:
            del self.products[exch]
            self.remove_exchange(exch)

    def get_exchange(self, exch: str) -> BaseExchange | None:
        return self.exchanges.get(exch.upper(), None)

    def add_exchange(self, exch: str) -> BaseExchange:
        exch = exch.upper()
        assert exch in SUPPORTED_CRYPTO_EXCHANGES, f'exchange {exch} is not supported'
        if not (exchange := self.get_exchange(exch)):
            Exchange = getattr(importlib.import_module(f'pfund.exchanges.{exch.lower()}.exchange'), 'Exchange')
            exchange = Exchange(self.env)
            self.exchanges[exch] = exchange
            self.connection_manager.add_api(exchange._ws_api)
            self.logger.debug(f'added {exch=}')
        return exchange
    
    def remove_exchange(self, exch: str):
        exch = exch.upper()
        if exch in self.exchanges:
            exchange = self.exchanges[exch]
            del self.exchanges[exch]
            self.connection_manager.remove_api(exchange._ws_api)
            self.logger.debug(f'removed {exch=}')
    
    def add_balance(self, exch: str, acc: str, ccy: str) -> CryptoBalance:
        exch, acc, ccy = convert_to_uppercases(exch, acc, ccy)
        if not (balance := self.get_balances(exch, acc=acc, ccy=ccy)):
            self.add_exchange(exch)
            account = self.get_account(exch, acc)
            balance = CryptoBalance(account, ccy)
            self.portfolio_manager.add_balance(balance)
            self.logger.debug(f'added {balance=}')
        return balance

    def add_position(self, exch: str, acc: str, pdt: str) -> CryptoPosition:
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
            for exch in self.accounts:
                for acc in self.accounts[exch]:
                    self.get_orders(exch, acc, is_api_call=True)
        func = inspect.stack()[0][3]
        Thread(target=work, name=func+'_thread', daemon=True).start()

    def get_orders(self, exch: str, acc: str, pdt: str='', oid: str='', eoid: str='', is_api_call=False, **kwargs) -> dict | None:
        exch, acc, pdt = convert_to_uppercases(exch, acc, pdt)
        if not is_api_call:
            return self.order_manager.get_orders(exch, acc, pdt=pdt, oid=oid, eoid=eoid)
        else:
            exchange = self.get_exchange(exch)
            account = self.get_account(exch, acc)
            return exchange.get_orders(account, pdt=pdt, **kwargs)

    def reconcile_trades(self):
        def work():
            for exch in self.accounts:
                for acc in self.accounts[exch]:
                    self.get_trades(exch, acc, is_api_call=True)
        func = inspect.stack()[0][3]
        Thread(target=work, name=func+'_thread', daemon=True).start()

    def get_trades(self, exch: str, acc: str, pdt: str='', is_api_call=False, **kwargs) -> dict | None:
        exch, acc, pdt = convert_to_uppercases(exch, acc, pdt)
        if not is_api_call:
            return self.order_manager.get_trades(...)
        else:
            exchange = self.get_exchange(exch)
            account = self.get_account(exch, acc)
            return exchange.get_trades(account, pdt=pdt, **kwargs)

    def reconcile_balances(self):
        def work():
            for exch in self.accounts:
                for acc in self.accounts[exch]:
                    self.get_balances(exch, acc, is_api_call=True)
        func = inspect.stack()[0][3]
        Thread(target=work, name=func+'_thread', daemon=True).start()

    def get_balances(self, exch: str, acc: str='', ccy: str='', is_api_call=False, **kwargs) -> dict | None:
        exch, acc, ccy = convert_to_uppercases(exch, acc, ccy)
        if not is_api_call:
            return self.portfolio_manager.get_balances(exch, acc, ccy=ccy)
        else:
            exchange = self.get_exchange(exch)
            account = self.get_account(exch, acc)
            return exchange.get_balances(account, ccy=ccy, **kwargs)

    def reconcile_positions(self):
        def work():
            for exch in self.accounts:
                for acc in self.accounts[exch]:
                    self.get_positions(exch, acc, is_api_call=True)
        func = inspect.stack()[0][3]
        Thread(target=work, name=func+'_thread', daemon=True).start()

    def get_positions(self, exch: str, acc: str='', pdt: str='', is_api_call=False, **kwargs) -> dict | None:
        exch, acc, pdt = convert_to_uppercases(exch, acc, pdt)
        if not is_api_call:
            return self.portfolio_manager.get_positions(exch, acc=acc, pdt=pdt)
        else:
            exchange = self.get_exchange(exch)
            account = self.get_account(exch, acc)
            return exchange.get_positions(account, pdt=pdt, **kwargs)

    def create_order(self, account: CryptoAccount, product: CryptoProduct, side: int, quantity: float, price=None, **kwargs):
        return CryptoOrder(account, product, side, qty=quantity, px=price, **kwargs)
    
    def place_orders(self, account: CryptoAccount, product: CryptoProduct, orders: list[BaseOrder]):
        exchange = self.get_exchange(account.exch)
        # TODO
        # self.rm checks risk

        num_orders = 0
        for o in orders:
            self.om.on_submitted(o)
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
            self.om.on_cancel(o)
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
            self.om.on_amend(o)