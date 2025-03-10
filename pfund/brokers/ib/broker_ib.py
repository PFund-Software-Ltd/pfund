"""This is a broker class for Interactive Brokers.
Conceptually, this is a combination of broker_crypto.py + exchange_base.py in crypto version
"""
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund.typing.data_kwargs import QuoteDataKwargs, TickDataKwargs, BarDataKwargs

from pfund.adapter import Adapter
from pfund.products import IBProduct
from pfund.accounts import IBAccount
from pfund.orders import IBOrder
from pfund.positions import IBPosition
from pfund.balances import IBBalance
from pfund.utils.utils import convert_to_uppercases
from pfund.brokers.broker_live import LiveBroker
from pfund.brokers.ib.ib_api import IBApi
from pfund.const.enums import PublicDataChannel, PrivateDataChannel


class IB_Broker(LiveBroker):
    def __init__(self, env: str, **configs):
        super().__init__(env, 'IB', **configs)
        config_path = f'{PROJ_CONFIG_PATH}/{self.bkr.value.lower()}'
        self.configs = Configuration(config_path, 'config')
        self.adapter = Adapter(config_path, self.configs.load_config_section('adapter'))
        self.account = None
        
        # API
        self._api = IBApi(self.env, self.adapter)
        self.connection_manager.add_api(self._api)

    def start(self, zmq=None):
        super().start(zmq=zmq)

    # EXTEND
    @staticmethod
    def derive_exch(pdt: str):
        bccy, qccy, ptype, *args = IBProduct.parse_product_name(pdt)
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

    def add_channel(self, channel: PublicDataChannel | PrivateDataChannel, type_, **kwargs):
        if type_.lower() == 'public':
            assert 'product' in kwargs, 'Keyword argument "product" is missing'
            if channel == PublicDataChannel.kline:
                assert 'period' in kwargs and 'timeframe' in kwargs, 'Keyword arguments "period" or/and "timeframe" is missing'
        elif type_.lower() == 'private':
            assert 'account' in kwargs, 'Keyword argument "account" is missing'
        self._api.add_channel(channel, type_, **kwargs)

    # TODO
    def add_custom_data(self):
        pass
    
    def add_data(
        self, 
        product: str, 
        resolutions: list[str] | str, 
        resamples: dict[str, str] | None=None,
        auto_resample=None,  # FIXME
        quote_data: dict | QuoteDataKwargs | None=None,
        tick_data: dict | TickDataKwargs | None=None,
        bar_data: dict | BarDataKwargs | None=None,
        exch: str='', 
        **product_specs
    ):
        '''
        Args:
            product: product basis, defined as {base_asset}_{quote_asset}_{product_type}, e.g. BTC_USDT_PERP
        '''
        exch = exch or self.derive_exch(product)
        exch, product_basis = exch.upper(), product.upper()
        product: IBProduct = self.add_product(exch, product_basis, **product_specs)
        datas = self.data_manager.add_data(
            product, 
            resolutions, 
            resamples=resamples,
            auto_resample=auto_resample,
            quote_data=quote_data,
            tick_data=tick_data,
            bar_data=bar_data,
        )
        for data in datas:
            self.add_data_channel(data, **kwargs)
        return datas
    
    def add_data_channel(self, data, **kwargs):
        if data.is_time_based():
            if data.is_resamplee():
                return
            timeframe = data.timeframe
            if timeframe.is_quote():
                channel = PublicDataChannel.orderbook
            elif timeframe.is_tick():
                channel = PublicDataChannel.tradebook
            else:
                channel = PublicDataChannel.kline
            self.add_channel(channel, 'public', product=data.product, period=data.period, timeframe=str(timeframe), **kwargs)
        else:
            raise NotImplementedError
    
    def create_product(self, exch: str, pdt: str, **kwargs) -> IBProduct:
        bccy, qccy, ptype, *args = IBProduct.parse_product_name(pdt)
        product = IBProduct(exch, bccy, qccy, ptype, *args, **kwargs)
        return product

    def get_product(self, pdt: str, exch: str='') -> IBProduct | None:
        exch = exch or self.derive_exch(pdt)
        return self._products[exch.upper()].get(pdt.upper(), None)

    def add_product(self, exch: str, pdt: str, **kwargs) -> IBProduct:
        exch, pdt = exch.upper(), pdt.upper()
        if not (product := self.get_product(exch=exch, pdt=pdt)):
            product = self.create_product(exch, pdt, **kwargs)
            self._products[exch][product.name] = product
            self._api.add_product(product, **kwargs)
            self.logger.debug(f'added product {product.name}')
        return product

    def get_account(self, acc: str) -> IBAccount | None:
        return self._accounts[self.bkr].get(acc.upper(), None)

    def add_account(self, host: str='', port: int=None, client_id: int=None, name: str='', **kwargs) -> IBAccount:
        if not (account := self.get_account(name)):
            account = IBAccount(self.env, host=host, port=port, client_id=client_id, name=name, **kwargs)
            self._accounts[self.bkr][account.name] = account
            self.account = account
            self._api.add_account(account)
            self.logger.debug(f'added {account=}')
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
            self.portfolio_manager.add_balance(balance)
            self.logger.debug(f'added {balance=}')
        return balance
    
    def add_position(self, exch: str, acc: str, pdt: str) -> IBPosition | None:
        exch, acc, pdt = convert_to_uppercases(exch, acc, pdt)
        if not (position := self.get_positions(exch=exch, acc=acc, pdt=pdt)):
            account = self.get_account(acc)
            product = self.add_product(exch, pdt)
            position = IBPosition(account, product)
            self.portfolio_manager.add_position(position)
            self.logger.debug(f'added {position=}')
        return position

    def add_order(self, exch: str, acc: str, pdt: str) -> IBOrder | None:
        exch, acc, pdt = convert_to_uppercases(exch, acc, pdt)
        if not (order := self.get_orders(acc)):
            product = self.add_product(exch, pdt)
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
        account = self.get_account(acc)
        product = self.add_product(exch, pdt)    
        return IBOrder(account, product, *args, **kwargs)
    
    def place_order(self, o):
        self.om.on_submitted(o)
        self._api.placeOrder(o.orderId, o.contract, o)