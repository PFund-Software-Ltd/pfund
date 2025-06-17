from __future__ import annotations
from typing import Literal, TYPE_CHECKING, overload, TypeAlias
if TYPE_CHECKING:
    from pfund.typing import (
        StrategyT, 
        tTradingVenue, 
        tCryptoExchange,
        ProductKey,
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
from pfund.mixins.trade_mixin import TradeMixin
from pfund.enums import TradingVenue, RunMode
from pfund.proxies.actor_proxy import ActorProxy


class BaseStrategy(TradeMixin, ABC, metaclass=MetaStrategy):    
    def __init__(self, *args, **kwargs):
        self.positions: dict[AccountName, dict[ProductKey, BasePosition]] = {}
        self.balances: dict[AccountName, dict[Currency, BaseBalance]] = {}
        # NOTE: includes submitted orders and opened orders
        self.orders: dict[AccountName, list[BaseOrder]] = {}
        # TODO: create Trade object, BaseTrade?
        # self.trades: dict[AccountName, list[BaseTrade]] = {}
        
        self.strategies: dict[str, BaseStrategy] = {}

        self.__mixin_post_init__(*args, **kwargs)  # calls TradeMixin.__mixin_post_init__()
    
    @abstractmethod
    def backtest(self, df: data_tool_backtest.BacktestDataFrame):
        pass
    
    # TODO:
    def add_risk_guard(self, risk_guard: RiskGuard):
        self._assert_not_frozen()
        raise NotImplementedError("RiskGuard is not implemented yet")
    
    def is_sub_strategy(self) -> bool:
        return bool(self._consumer)
    
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
    ) -> SimulatedAccount: ...
    
    def add_account(self, trading_venue: tTradingVenue, name: str='', **kwargs) -> BaseAccount:
        self._assert_not_frozen()
        trading_venue = TradingVenue[trading_venue.upper()]
        account: BaseAccount = self._engine._register_account(trading_venue, name=name, **kwargs)
        self.positions[account.name] = {}
        self.balances[account.name] = {}
        self.orders[account.name] = []
        # TODO make maxlen a variable in config
        # self.trades[account.name] = deque(maxlen=10)
        self.logger.debug(f'added {trading_venue} account "{account.name}"')
        return account
    
    def add_strategy(self, strategy: StrategyT, name: str='') -> StrategyT:
        self._assert_not_frozen()
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
    
    def start(self):
        # TODO: check if e.g. exchange balances and positions are ready, if backfilling is finished
        super().start()
        self.add_strategies()
        # TODO: start components
        # for strategy in self.strategies.values():
        #     strategy.start()
        if self._run_mode == RunMode.REMOTE:
            self._databoy._setup_messaging()
            self._databoy._collect()
        
    def stop(self, reason: str=''):
        super().stop(reason=reason)

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
    