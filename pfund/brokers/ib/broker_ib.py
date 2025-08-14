"""This is a broker class for Interactive Brokers.
Conceptually, this is the equivalent of broker_crypto.py + exchange_base.py in crypto
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Literal
if TYPE_CHECKING:
    from pfund._typing import tEnvironment, FullDataChannel, AccountName, ProductName
    from pfund.datas.data_time_based import TimeBasedData

from pfund.adapter import Adapter
from pfund.products.product_ib import IBProduct
from pfund.accounts.account_ib import IBAccount
from pfund.orders.order_ib import IBOrder
from pfund.positions.position_ib import IBPosition
from pfund.balances.balance_ib import IBBalance
from pfund.utils.utils import convert_to_uppercases
from pfund.brokers.broker_base import BaseBroker
from pfund.brokers.ib.ib_api import IBAPI
from pfund.enums import PublicDataChannel, PrivateDataChannel, Environment, Broker


class IBBroker(BaseBroker):
    name = Broker.IB
    adapter = Adapter(name)
    
    def __init__(self, env: Environment | tEnvironment=Environment.SANDBOX):
        super().__init__(env=env)
        # FIXME: check if only supports one account
        self.account = None
        self._accounts: dict[Literal[Broker.IB], dict[AccountName, IBAccount]] = { self.name: {} }
        self._api = IBAPI(self._env)

    @property
    def accounts(self):
        return self._accounts[self.name]
    
    def _add_default_private_channels(self):
        for channel in list(PrivateDataChannel.__members__) + ['account_update', 'account_summary']:
            self.add_private_channel(channel)
    
    def add_public_channel(self, channel: PublicDataChannel | FullDataChannel, data: TimeBasedData | None=None):
        if channel.lower() in PublicDataChannel.__members__:
            assert data is not None, 'data object is required for public channels'
            channel: FullDataChannel = self._api._create_public_channel(data.product, data.resolution)
        self._api.add_channel(channel, channel_type='public')
    
    def add_private_channel(self, channel: PrivateDataChannel | FullDataChannel):
        if channel.lower() in PrivateDataChannel.__members__:
            channel: FullDataChannel = self._api._create_private_channel(channel)
        self._api.add_channel(channel, channel_type='private')

    def get_account(self, name: AccountName) -> IBAccount:
        return self.accounts[name]

    def add_account(self, name: AccountName='', host: str='', port: int | None=None, client_id: int | None=None) -> IBAccount:
        if name not in self.accounts:
            account = IBAccount(env=self._env, name=name, host=host, port=port, client_id=client_id)
            self.accounts[account.name] = account
            self.account = account
            self._api.add_account(account)
        else:
            raise ValueError(f'account name {name} has already been added')
            # FIXME
            # if account.name != name.upper():
            #     raise Exception(f'Only one primary account is supported and account {self.account} is already set up')
        return account
    
    def get_product(self, name: ProductName, exch: str='') -> IBProduct:
        if exch:
            return self._products[exch.upper()][name]
        else:
            products = [_name for _exch in self._products for _name in self._products[_exch] if _name == name]
            if len(products) == 1:
                return products[0]
            else:
                raise ValueError(f'product name {name} has multiple products across exchanges, please specify `exch`')
    
    def add_product(self, basis: str, exch: str='', name: ProductName='', symbol: str='', **specs) -> IBProduct:
        exch = exch.upper() or self._derive_exchange(basis)
        product: IBProduct = self.create_product(basis, exch=exch, name=name, symbol=symbol, **specs)
        if product.name not in self._products[exch]:
            # TODO: load market configs
            # market_configs = self.load_market_configs()
            # if product.symbol not in market_configs[product.category]:
            #     raise ValueError(
            #         f"The symbol '{product.symbol}' is not found in the market configurations. "
            #         f"It might be delisted, or your market configurations could be outdated. "
            #         f"Please set 'refetch_market_configs=True' in TradeEngine's settings to refetch the latest market configurations."
            #     )
            self._products[exch][product.name] = product
            self._api.add_product(product, **specs)
            self.adapter.add_mapping(str(product.type), product.name, product.symbol)
        else:
            existing_product: IBProduct = self.get_product(product.name, exch=exch)
            # assert products are the same with the same name
            if existing_product == product:
                product = existing_product
            else:
                raise ValueError(f'product name {name} has already been used for {existing_product}')
        return product

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