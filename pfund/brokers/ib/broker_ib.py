"""This is a broker class for Interactive Brokers.
Conceptually, this is the equivalent of broker_crypto.py + exchange_base.py in crypto
"""
from __future__ import annotations
from typing import Literal, TYPE_CHECKING

from pfund.config import Configuration
if TYPE_CHECKING:
    from pfund.typing import tEnvironment
    from pfund.datas.data_base import BaseData
    from pfund.datas.data_config import DataConfig
    from pfund.datas.data_time_based import TimeBasedData

from pfund.adapter import Adapter
from pfund.products.brokers.product_ib import IBProduct
from pfund.accounts.account_ib import IBAccount
from pfund.orders.order_ib import IBOrder
from pfund.positions.position_ib import IBPosition
from pfund.balances.balance_ib import IBBalance
from pfund.utils.utils import convert_to_uppercases
from pfund.brokers.broker_trade import TradeBroker
from pfund.brokers.ib.ib_api import IBApi
from pfund.enums import PublicDataChannel, PrivateDataChannel, DataChannelType


class IBBroker(TradeBroker):
    def __init__(self, env: tEnvironment='SANDBOX', **configs):
        super().__init__(env, 'IB', **configs)
        config_path = f'{PROJ_CONFIG_PATH}/{self._name.value.lower()}'
        self.configs = Configuration(config_path, 'config')
        self.adapter = Adapter(config_path, self.configs.load_config_section('adapter'))
        self.account = None
        
        # API
        self._api = IBApi(self._env, self.adapter)
        self._connection_manager.add_api(self._api)

    @property
    def accounts(self):
        return self._accounts[self._name]
    
    def start(self, zmq=None):
        super().start(zmq=zmq)

    # EXTEND
    @staticmethod
    def _derive_exch(product_basis: str):
        bccy, qccy, ptype, *args = IBProduct.parse_product_name(product_basis)
        if ptype == 'FX':
            exch = 'IDEALPRO'
        elif ptype == 'CRYPTO':
            raise Exception(f'when product type is {ptype}, `exch` must be provided in add_data(exch=...)')
        else:
            exch = 'SMART'
        return exch

    # EXTEND
    @staticmethod
    def _standardize_ptype(ptype: str):
        if ptype in ['CASH', 'CURRENCY', 'FX', 'FOREX', 'SPOT']:
            ptype = 'FX'
        elif ptype in ['CRYPTO', 'CRYPTOCURRENCY']:
            ptype = 'CRYPTO'
        elif ptype in ['FUT', 'FUTURE']:
            ptype = 'FUT'
        elif ptype in ['OPT', 'OPTION']:
            ptype = 'OPT'
        return ptype

    def add_channel(
        self, 
        channel: PublicDataChannel | PrivateDataChannel | str, 
        channel_type: DataChannelType,
        data: BaseData | None=None,
        **kwargs
    ):
        if type_.lower() == 'public':
            assert 'product' in kwargs, 'Keyword argument "product" is missing'
            if channel == PublicDataChannel.candlestick:
                assert 'period' in kwargs and 'timeframe' in kwargs, 'Keyword arguments "period" or/and "timeframe" is missing'
        elif type_.lower() == 'private':
            assert 'account' in kwargs, 'Keyword argument "account" is missing'
        self._api.add_channel(channel, type_, **kwargs)

    def add_data_channel(self, data: TimeBasedData, **kwargs):
        if data.is_time_based():
            if data.is_resamplee():
                return
            timeframe = data.timeframe
            if timeframe.is_quote():
                channel = PublicDataChannel.orderbook
            elif timeframe.is_tick():
                channel = PublicDataChannel.tradebook
            else:
                channel = PublicDataChannel.candlestick
            self.add_channel(channel, 'public', product=data.product, period=data.period, timeframe=str(timeframe), **kwargs)
        else:
            raise NotImplementedError
    
    def create_product(self, exch: str, pdt: str, symbol: str='', **kwargs) -> IBProduct:
        bccy, qccy, ptype, *args = IBProduct.parse_product_name(pdt)
        product = IBProduct(exch, bccy, qccy, ptype, *args, **kwargs)
        return product

    def get_product(self, pdt: str, exch: str='') -> IBProduct | None:
        exch = exch or self._derive_exch(pdt)
        return self._products[exch.upper()].get(pdt.upper(), None)

    def add_product(self, basis: str, exch: str='', name: str='', symbol: str='', **specs) -> IBProduct:
        basis = basis.upper()
        exch = exch or self._derive_exch(basis)
        if not (product := self.get_product(exch=exch, pdt=basis)):
            product = self.create_product(exch, basis, symbol=symbol, **specs)
            self._products[exch][str(product)] = product
            self._api.add_product(product, **specs)
        return product

    def get_account(self, acc: str) -> IBAccount | None:
        return self.accounts.get(acc.upper(), None)

    def _create_account(self, name: str, host: str, port: int | None, client_id: int | None) -> IBAccount:
        return IBAccount(env=self._env, name=name, host=host, port=port, client_id=client_id)

    def add_account(
        self, 
        name: str='', 
        host: str='', 
        port: int | None=None, 
        client_id: int | None=None, 
    ) -> IBAccount:
        if not (account := self.get_account(name)):
            account = self._create_account(
                name=name, 
                host=host, 
                port=port, 
                client_id=client_id,
            )
            self.accounts[account.name] = account
            self.account = account
            self._api.add_account(account)
            self._logger.debug(f'added {account=}')
        else:
            # TODO
            if account.name != name.upper():
                raise Exception(f'Only one primary account is supported and account {self.account} is already set up')
        return account

    def add_balance(self, acc: str, ccy: str) -> IBBalance | None:
        acc, ccy = convert_to_uppercases(acc, ccy)
        if not (balance := self.get_balances(acc=acc, ccy=ccy)):
            account = self.get_account(acc)
            balance = IBBalance(account, ccy)
            self._portfolio_manager.add_balance(balance)
            self._logger.debug(f'added {balance=}')
        return balance
    
    def add_position(self, exch: str, acc: str, pdt: str) -> IBPosition | None:
        exch, acc, pdt = convert_to_uppercases(exch, acc, pdt)
        if not (position := self.get_positions(exch=exch, acc=acc, pdt=pdt)):
            account = self.get_account(acc)
            product = self.add_product(exch, pdt)
            position = IBPosition(account, product)
            self._portfolio_manager.add_position(position)
            self._logger.debug(f'added {position=}')
        return position

    def add_order(self, exch: str, acc: str, pdt: str) -> IBOrder | None:
        exch, acc, pdt = convert_to_uppercases(exch, acc, pdt)
        if not (order := self.get_orders(acc)):
            product = self.add_product(exch, pdt)
            order = IBOrder(self._env, acc, product)
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
            acc = self.account.name
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
        account = self.get_account(acc)
        product = self.add_product(exch, pdt)    
        return IBOrder(account, product, *args, **kwargs)
    
    def place_order(self, o):
        self._order_manager.on_submitted(o)
        self._api.placeOrder(o.orderId, o.contract, o)