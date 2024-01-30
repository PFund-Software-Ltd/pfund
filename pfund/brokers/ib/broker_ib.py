"""This is a broker class for Interactive Brokers.
Conceptually, this is a combination of broker_crypto.py + exchange_base.py in crypto version
"""
from collections import defaultdict

from pfund.adapter import Adapter
from pfund.config.configuration import Configuration
from pfund.const.commons import SUPPORTED_PRODUCT_TYPES
from pfund.products import IBProduct
from pfund.accounts import IBAccount
from pfund.orders import IBOrder
from pfund.positions import IBPosition
from pfund.balances import IBBalance
from pfund.utils.utils import convert_to_uppercases
from pfund.brokers.broker_live import LiveBroker
from pfund.brokers.ib.ib_api import IBApi


class IBBroker(LiveBroker):
    def __init__(self, env, **configs):
        super().__init__(env, 'IB', **configs)
        self.configs = Configuration(self.bkr, 'config')
        self.adapter = Adapter(self.bkr, self.configs.load_config_section('adapter'))
        self.account = None
        
        # API
        self._api = IBApi(self.env, self.adapter)
        self.connection_manager.add_api(self._api)

    def start(self, zmq=None):
        super().start(zmq=zmq)

    @staticmethod
    # EXTEND
    def derive_exch(bccy: str, qccy: str, ptype: str):
        if ptype == 'CASH':
            exch = 'IDEALPRO'
        elif ptype == 'CRYPTO':
            raise Exception(f'when product type is {ptype}, `exch` must be provided in add_data(exch=...)')
        else:
            exch = 'SMART'
        return exch

    @staticmethod
    # EXTEND
    def _standardize_ptype(ptype: str):
        if ptype in ['CASH', 'CURRENCY', 'FX', 'FOREX', 'SPOT']:
            ptype = 'CASH'
        elif ptype in ['CRYPTO', 'CRYPTOCURRENCY']:
            ptype = 'CRYPTO'
        elif ptype in ['FUT', 'FUTURE']:
            ptype = 'FUT'
        elif ptype in ['OPT', 'OPTION']:
            ptype = 'OPT'
        return ptype

    def add_channel(self, channel, type_, **kwargs):
        if type_.lower() == 'public':
            assert 'product' in kwargs, 'Keyword argument "product" is missing'
            if channel == 'kline':
                assert 'period' in kwargs and 'timeframe' in kwargs, 'Keyword arguments "period" or/and "timeframe" is missing'
        elif type_.lower() == 'private':
            assert 'account' in kwargs, 'Keyword argument "account" is missing'
        self._api.add_channel(channel, type_, **kwargs)

    # TODO
    def add_custom_data(self):
        pass
    
    def add_data(self, base_currency, quote_currency, product_type, *args, **kwargs):
        base_currency, quote_currency, product_type = convert_to_uppercases(base_currency, quote_currency, product_type)
        product_type = self._standardize_ptype(product_type)
        if 'exch' in kwargs:
            exch = kwargs['exch'].upper()
            del kwargs['exch']
        else:
            exch = self.derive_exch(base_currency, quote_currency, product_type)
        product = self.add_product(exch, base_currency, quote_currency, product_type, *args, **kwargs)
        datas = self.data_manager.add_data(product, **kwargs)
        for data in datas:
            self.add_data_channel(data, **kwargs)
        return datas
    
    def add_data_channel(self, data, **kwargs):
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
            self.add_channel(channel, 'public', product=data.product, period=data.period, timeframe=str(timeframe), **kwargs)
        else:
            raise NotImplementedError

    def get_product(self, pdt: str, exch: str='') -> IBProduct | None:
        if not exch:
            bccy, qccy, ptype, *args = IBProduct.parse_product_name(pdt)
            exch = self.derive_exch(bccy, qccy, ptype)
        return self.products[exch.upper()].get(pdt.upper(), None)

    def add_product(self, exch, bccy, qccy, ptype, *args, **kwargs):
        assert ptype.upper() in SUPPORTED_PRODUCT_TYPES, f'{self.bkr} product type {ptype} is not supported, {SUPPORTED_PRODUCT_TYPES=}'
        pdt = IBProduct.create_product_name(bccy, qccy, ptype, *args, **kwargs)
        if not (product := self.get_product(exch=exch, pdt=pdt)):
            product = IBProduct(exch, bccy, qccy, ptype, *args, **kwargs)
            self.products[exch][product.name] = product
            self._api.add_product(product, **kwargs)
            self.logger.debug(f'added product {product.name}')
        return product

    def get_account(self) -> IBAccount | None:
        return self.account

    def add_account(self, host: str='', port: int=None, client_id: int=None, acc: str='', **kwargs) -> IBAccount:
        if not (account := self.get_account()):
            account = IBAccount(self.env, host=host, port=port, client_id=client_id, acc=acc, **kwargs)
            self.accounts['IB'] = account
            self.account = account
            self._api.add_account(account)
            self.logger.debug(f'added {account=}')
        else:
            # TODO
            if account.name != acc.upper():
                raise Exception(f'Only one primary account is supported and account {self.account} is already set up')
        return account

    def add_balance(self, acc: str, ccy: str) -> IBBalance | None:
        acc, ccy = convert_to_uppercases(acc, ccy)
        if not (balance := self.get_balances(acc=acc, ccy=ccy)):
            account = self.get_account()
            balance = IBBalance(account, ccy)
            self.portfolio_manager.add_balance(balance)
            self.logger.debug(f'added {balance=}')
        return balance
    
    def add_position(self, exch: str, acc: str, pdt: str) -> IBPosition | None:
        exch, acc, pdt = convert_to_uppercases(exch, acc, pdt)
        if not (position := self.get_positions(exch=exch, acc=acc, pdt=pdt)):
            bccy, qccy, ptype, *args = IBProduct.parse_product_name(pdt)
            account = self.get_account()
            product = self.add_product(exch, bccy, qccy, ptype, *args)
            position = IBPosition(account, product)
            self.portfolio_manager.add_position(position)
            self.logger.debug(f'added {position=}')
        return position

    def add_order(self, exch: str, acc: str, pdt: str) -> IBOrder | None:
        exch, acc, pdt = convert_to_uppercases(exch, acc, pdt)
        if not (order := self.get_orders(acc)):
            bccy, qccy, ptype, *args = IBProduct.parse_product_name(pdt)
            product = self.add_product(exch, bccy, qccy, ptype, *args)
            order = IBOrder(self.env, acc, product)
            self.orders[acc][order.oid] = order
        return order
    
    # TODO
    def get_orders(self, acc: str='', pdt: str='') -> dict | IBOrder:
        """Gets orders from an IB account.
        Account name `acc` will be automatically filled using the primary account if not provided.
        Therefore, `acc` is always non-empty
        Case 1: empty `exch` and empty `pdt`
            returns positions for that specific account
        Case 2: empty `exch` and non-empty `pdt`
            returns positions from different exchanges with the same product
        Case 3: non-empty `exch` and empty `pdt`
            returns positions in that specific exchange
        Case 4: non-empty `exch` and non-empty `pdt`
            returns position in that specific exchange for that specific product
        
        Args:
            acc: account name. If empty, use primary account by default.
            exch: exchange name.
            pdt: product name.
        """
        return orders

    def get_balances(self, acc: str='', ccy: str='') -> dict | IBBalance:
        """Gets balances from an IB account.
        Account name `acc` will be automatically filled using the primary account if not provided.
        Therefore, `acc` is always non-empty
        Case 1: empty `ccy`
            returns balances for that specific account
        Case 2: non-empty `ccy`
            returns balance for that specific currency
        
        Args:
            acc: account name. If empty, use primary account by default.
            ccy: currency name.
        """
        acc, ccy = convert_to_uppercases(acc, ccy)
        if not acc:
            acc = self.account.acc
        balances = self.balances[acc]
        if ccy:
            balances = balances[ccy]
        return balances

    def get_positions(self, exch: str='', acc: str='', pdt: str='') -> dict | IBPosition:
        """Gets positions from an IB account.
        Account name `acc` will be automatically filled using the primary account if not provided.
        Therefore, `acc` is always non-empty
        Case 1: empty `exch` and empty `pdt`
            returns positions for that specific account
        Case 2: empty `exch` and non-empty `pdt`
            returns positions from different exchanges with the same product
        Case 3: non-empty `exch` and empty `pdt`
            returns positions in that specific exchange
        Case 4: non-empty `exch` and non-empty `pdt`
            returns position in that specific exchange for that specific product
        
        Args:
            acc: account name. If empty, use primary account by default.
            exch: exchange name.
            pdt: product name.
        """
        exch, acc, pdt = convert_to_uppercases(exch, acc, pdt)
        # FIXME, positions should be acc -> pdt -> exch? havan't decided yet
        if not acc:
            acc = self.account.acc
        positions = self.positions[acc]
        if not exch:
            if pdt:
                positions = {_exch: position for _exch in positions for _pdt, position in positions[_exch].items() if pdt == _pdt}
        else:
            positions = self.positions[acc][exch]
            if pdt in positions:
                positions = positions[pdt]
        return positions
    
    def create_order(self, exch, acc, pdt, *args, **kwargs):
        account = self.get_account()
        bccy, qccy, ptype, *args = IBProduct.parse_product_name(pdt)
        product = self.add_product(exch, bccy, qccy, ptype, *args)    
        return IBOrder(account, product, *args, **kwargs)
    
    def place_order(self, o):
        self.om.on_submitted(o)
        self._api.placeOrder(o.orderId, o.contract, o)