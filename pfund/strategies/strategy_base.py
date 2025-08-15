from __future__ import annotations
from typing import Literal, TYPE_CHECKING, overload
if TYPE_CHECKING:
    from pfund._typing import (
        StrategyT, 
        tTradingVenue, 
        tCryptoExchange,
        ProductName,
        AccountName,
        Currency,
    )
    from pfund.enums import OrderSide
    from pfund.products.product_base import BaseProduct
    from pfund.positions.position_base import BasePosition
    from pfund.positions.position_crypto import CryptoPosition
    from pfund.positions.position_ib import IBPosition
    from pfund.balances.balance_base import BaseBalance
    from pfund.balances.balance_crypto import CryptoBalance
    from pfund.balances.balance_ib import IBBalance
    from pfund.accounts.account_base import BaseAccount
    from pfund.accounts.account_crypto import CryptoAccount
    from pfund.accounts.account_ib import IBAccount
    from pfund.accounts.account_simulated import SimulatedAccount
    from pfund.orders.order_base import BaseOrder
    from pfund.datas.data_base import BaseData
    from pfund.risk_guard import RiskGuard
    from pfund.data_tools import data_tool_backtest

from collections import deque
from abc import ABC, abstractmethod

from pfund.strategies.strategy_meta import MetaStrategy
from pfund.mixins.component_mixin import ComponentMixin
from pfund.enums import TradingVenue, Broker
from pfund.proxies.actor_proxy import ActorProxy


class BaseStrategy(ComponentMixin, ABC, metaclass=MetaStrategy):    
    def __init__(self, *args, **kwargs):       
        # TODO: also include sub-strategies' accounts
        self.accounts: dict[AccountName, BaseAccount] = {}
        self.strategies: dict[str, BaseStrategy] = {}
        # TODO: Portfolio (from pfolio?) at this level?
        # self._portfolio: Portfolio = ...
        # TODO: aggregate sub-strategies' portfolios
        # self._aggregated_portfolio: Portfolio = ...
        # self._risk_guards: list[RiskGuard] = []
        self._is_top_strategy: bool = False
        
        # FIXME: move positions, balances, orders, all to account object!
        # TODO: if its a sub-strategy, it should not have positions, balances, orders, etc.
        self.positions: dict[AccountName, dict[ProductName, BasePosition]] = {}
        self.balances: dict[AccountName, dict[Currency, BaseBalance]] = {}
        # NOTE: includes submitted orders and opened orders
        self.orders: dict[AccountName, list[BaseOrder]] = {}
        # TODO: create Trade object, BaseTrade?
        # self.trades: dict[AccountName, list[BaseTrade]] = {}
        
        self.__mixin_post_init__(*args, **kwargs)  # calls ComponentMixin.__mixin_post_init__()
    
    @abstractmethod
    def backtest(self, df: data_tool_backtest.BacktestDataFrame):
        pass
    
    # TODO: warning if sub-strategy adds risk guard
    def add_risk_guard(self, risk_guard: RiskGuard):
        raise NotImplementedError("RiskGuard is not implemented yet")
    
    def _set_top_strategy(self, is_top_strategy: bool):
        self._is_top_strategy = is_top_strategy
    
    def is_top_strategy(self) -> bool:
        return self._is_top_strategy
    
    def is_sub_strategy(self) -> bool:
        return not self._is_top_strategy
    
    def get_strategy(self, name: str) -> BaseStrategy | ActorProxy:
        return self.strategies[name]
    
    @overload
    def get_position(self, account: CryptoAccount, pdt: str) -> CryptoPosition | None: ...
    
    @overload
    def get_position(self, account: IBAccount, pdt: str) -> IBPosition | None: ...
    
    def get_position(self, account: BaseAccount, pdt: str) -> BasePosition | None:
        return self.positions[account].get(pdt, None)
    
    @overload
    def get_balance(self, account: CryptoAccount, ccy: str) -> CryptoBalance | None: ...
    
    @overload
    def get_balance(self, account: IBAccount, ccy: str) -> IBBalance | None: ...
    
    def get_balance(self, account: BaseAccount, ccy: str) -> BaseBalance | None:
        return self.balances[account].get(ccy, None)
    
    def to_dict(self) -> dict:
        return {
            **super().to_dict(),
            'is_sub_strategy': self.is_sub_strategy(),
            'accounts': [repr(account) for account in self.accounts.values()],
            'strategies': [strategy.to_dict() for strategy in self.strategies.values()],
        }
    
    @overload
    def add_account(
        self, 
        trading_venue: tCryptoExchange, 
        name: str='', 
        key: str='', 
        secret: str='', 
    ) -> CryptoAccount: ...
    
    @overload
    def add_account(
        self,
        trading_venue: Literal['IB'],
        name: str='',
        host: str='',
        port: int | None=None,
        client_id: int | None=None,
    ) -> IBAccount: ...
    
    @overload
    def add_account(
        self,
        trading_venue: tTradingVenue,
        name: str='',
        initial_balances: dict[str, float] | None=None, 
        initial_positions: dict[BaseProduct, float] | None=None,
        **kwargs,
    ) -> SimulatedAccount: ...

    def _create_account(self, trading_venue: tTradingVenue, name: str='', **kwargs) -> BaseAccount:
        from pfund.brokers import create_broker
        # NOTE: broker is only used to create account but does nothing else
        broker = create_broker(env=self.env, bkr=TradingVenue[trading_venue.upper()].broker)
        if broker.name == Broker.CRYPTO:
            exch = trading_venue
            account =  broker.add_account(exch=exch, name=name or self.name, **kwargs)
        elif broker.name == Broker.IB:
            account = broker.add_account(name=name or self.name, **kwargs)
        else:
            raise NotImplementedError(f"Broker {broker.name} is not supported")
        return account
    
    def get_accounts(self) -> list[BaseAccount]:
        return list(self.accounts.values())
    
    def add_account(self, trading_venue: tTradingVenue, name: str='', **kwargs) -> BaseAccount:
        if self.is_sub_strategy():
            raise ValueError(f"Sub-strategy '{self.name}' cannot add accounts")
        trading_venue = TradingVenue[trading_venue.upper()]
        account: BaseAccount = self._create_account(trading_venue, name=name, **kwargs)
        self.accounts[account.name] = account
        self.positions[account.name] = {}
        self.balances[account.name] = {}
        self.orders[account.name] = []
        # TODO make maxlen a variable in config
        # self.trades[account.name] = deque(maxlen=10)
        self.logger.debug(f'added {trading_venue} account "{account.name}"')
        return account
    
    # FIXME: update, refer to or reuse _add_component() in component_mixin.py
    # TODO: _setup_logging, _set_engine etc.
    def add_strategy(self, strategy: StrategyT, name: str='') -> StrategyT:
        if strategy.is_top_strategy():
            raise ValueError(f"Top strategy '{self.name}' cannot be added as a sub-strategy")
        assert isinstance(strategy, BaseStrategy), \
            f"strategy '{strategy.__class__.__name__}' is not an instance of BaseStrategy. Please create your strategy using 'class {strategy.__class__.__name__}(pf.Strategy)'"
        if name:
            strategy._set_name(name)
        strategy._set_trading_store()
        strategy._create_logger()
        strategy._set_consumer(self)
        strategy._set_resolution(self.resolution)
        strat = strategy.name
        if strat in self.strategies:
            return self.strategies[strat]
        self.strategies[strat] = strategy
        self.logger.debug(f"added sub-strategy '{strat}'")
        return strategy
    
    def _update_positions(self, position):
        self.positions[position.account][position.pdt] = position
        self.on_position(position.account, position)

    def _update_balances(self, balance):
        self.balances[balance.account][balance.ccy] = balance
        self.on_balance(balance.account, balance)

    def _update_orders(self, order, type_: Literal['submitted', 'opened', 'closed', 'amended']):
        if type_ in ['submitted', 'opened']:
            if order not in self.orders[order.account]:
                self.orders[order.account].append(order)
        elif type_ == 'closed':
            if order in self.orders[order.account]:
                self.orders[order.account].remove(order)
        self.on_order(order.account, order, type_)

    def _update_trades(self, order, type_: Literal['partial', 'filled']):
        trade = order.trades[-1]
        self.trades[order.account].append(trade)
        self.on_trade(order.account, trade, type_)
        
    # TODO
    def _next(self, data: BaseData):
        # NOTE: only sub-strategies have predict()
        # pred_y = self.predict(X)
        pass
    
    def _gather(self):
        # TODO: check if e.g. exchange balances and positions are ready, if backfilling is finished?
        # TODO: top strategy must have an account
        self.add_strategies()
        self.add_accounts()
        super()._gather()
        
    def create_order(
        self, 
        account: BaseAccount, 
        product: BaseProduct, 
        side: OrderSide | Literal['BUY', 'SELL'] | Literal[1, -1],
        quantity: float, 
        price: float | None=None, 
        **kwargs
    ):
        broker = self.get_broker(account.bkr)
        return broker.create_order(creator=self.name, account=account, product=product, side=side, quantity=quantity, price=price, **kwargs)
    
    def place_orders(self, account: BaseAccount, product: BaseProduct, orders: list[BaseOrder] | BaseOrder):
        if not isinstance(orders, list):
            orders = [orders]
        broker = self.get_broker(account.bkr)
        broker.place_orders(account, product, orders)
        return orders

    def cancel_orders(self, account: BaseAccount, product: BaseProduct, orders: list[BaseOrder] | BaseOrder):
        if not isinstance(orders, list):
            orders = [orders]
        broker = self.get_broker(account.bkr)
        broker.cancel_orders(account, product, orders)
        return orders

    # TODO
    def cancel_all_orders(self):
        pass

    # TODO
    def amend_orders(self, bkr, exch='', acc=''):
        pass


    '''
    ************************************************
    Override Methods
    Override these methods in your subclass to implement your custom behavior.
    ************************************************
    '''
    def add_strategies(self):
        pass
    
    def add_accounts(self):
        pass
    
    def on_position(self, account, position):
        pass

    def on_balance(self, account, balance):
        pass

    def on_order(self, account, order, type_: Literal['submitted', 'opened', 'closed', 'amended']):
        pass

    def on_trade(self, account, trade: dict, type_: Literal['partial', 'filled']):
        pass


    '''
    ************************************************
    Sugar Functions
    ************************************************
    '''
    def buy(self, account, product, price, quantity, **kwargs):
        order = self.create_order(account, product, 1, quantity, price=price, o_type='LIMIT', **kwargs)
        self.place_orders(account, product, [order])
    buy_limit = buy_limit_order = buy
    
    def buy_market_order(self, account, product, quantity):
        order = self.create_order(account, product, 1, quantity, o_type='MARKET')
        self.place_orders(account, product, [order])
    buy_market = buy_market_order

    def sell(self, account, product, price, quantity, **kwargs):
        order = self.create_order(account, product, -1, quantity, price=price, o_type='LIMIT', **kwargs)
        self.place_orders(account, product, [order])
    sell_limit = sell_limit_order = sell

    def sell_market_order(self, account, product, quantity):
        order = self.create_order(account, product, -1, quantity, o_type='MARKET')
        self.place_orders(account, product, [order])
    sell_market = sell_market_order
    