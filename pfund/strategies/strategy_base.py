from __future__ import annotations
from typing import Literal, TYPE_CHECKING, overload
if TYPE_CHECKING:
    from pfeed.typing import tDATA_SOURCE
    from mtflow.stores.trading_store import TradingStore
    from pfund.typing import StrategyT, DataConfigDict, tTRADING_VENUE, tCRYPTO_EXCHANGE
    from pfund.brokers.broker_base import BaseBroker
    from pfund.products.product_base import BaseProduct
    from pfund.positions.position_base import BasePosition
    from pfund.positions.position_crypto import CryptoPosition
    from pfund.positions.position_ib import IBPosition
    from pfund.balances.balance_base import BaseBalance
    from pfund.balances.balance_crypto import CryptoBalance
    from pfund.balances.balance_ib import IBBalance
    from pfund.datas.data_time_based import TimeBasedData
    from pfund.accounts.account_base import BaseAccount
    from pfund.accounts.account_crypto import CryptoAccount
    from pfund.accounts.account_ib import IBAccount
    from pfund.accounts.account_simulated import SimulatedAccount
    from pfund.orders.order_base import BaseOrder
    from pfund.datas.data_base import BaseData
    from pfund.models.model_base import BaseModel, BaseFeature
    from pfund.indicators.indicator_base import BaseIndicator
    from pfund.datas.data_bar import Bar
    from pfund.data_tools.data_tool_base import BaseDataTool
    from pfund.risk_guard import RiskGuard
    from pfund.datas.data_config import DataConfig
    from pfund.data_tools import data_tool_backtest
    from pfund.datas.resolution import Resolution

from collections import defaultdict, deque
from abc import ABC, abstractmethod

from pfund.strategies.strategy_meta import MetaStrategy
from pfund.mixins.trade_mixin import TradeMixin
from pfund.enums import TradingVenue, Broker


class BaseStrategy(TradeMixin, ABC, metaclass=MetaStrategy):    
    def __init__(self, *args, **kwargs):
        from pfund.databoy import DataBoy
        
        cls = self.__class__
        cls.load_config()
        cls.load_params()
        
        self._args = args
        self._kwargs = kwargs
        self._name = self._get_default_name()
        
        self.logger = None
        self._engine = None
        self._store: TradingStore | None = None
        self._databoy = DataBoy()
        
        # FIXME: move to mtflow?
        # self._data_tool: BaseDataTool = self._create_data_tool()
        
        self._resolution: Resolution | None = None
        
        # TODO: move to mtflow
        self._consumer: BaseStrategy | None = None
        self._listeners: dict[BaseData, list[BaseStrategy | BaseModel]] = defaultdict(list)

        self.accounts = defaultdict(dict)  # {trading_venue: {acc1: account1, acc2: account2} }
        self.positions = {}  # {account: {pdt: position} }
        self.balances = {}  # {account: {ccy: balance}}
        # NOTE: includes submitted orders and opened orders
        self.orders = {}  # {account: [order, ...]}
        self.trades = {}  # {account: [trade, ...]}
        
        self.strategies: dict[str, BaseStrategy] = {}
        self.models: dict[str, BaseModel] = {}
        self.features: dict[str, BaseFeature] = {}
        self.indicators: dict[str, BaseIndicator] = {}
        # NOTE: current strategy's signal is consumer's prediction
        self.predictions = {}  # {strat/mdl: pred_y}
        self._signals = {}  # {data: signal}, for strategy, signal is buy/sell/null
        self._last_signal_ts = {}  # {data: ts}
        self._signal_cols = []
        self._num_signal_cols = 0

        self._strategy_signature = (args, kwargs)

        self._is_running = False
    
    @staticmethod
    def _get_pfund_strategy_class():
        '''Gets the actual strategy class under ray's ActorHandle'''
        return BaseStrategy.__pfund_strategy_class__
        
    @abstractmethod
    def backtest(self, df: data_tool_backtest.BacktestDataFrame):
        pass

    # TODO:
    def add_risk_guard(self, risk_guard: RiskGuard):
        raise NotImplementedError("RiskGuard is not implemented yet")
    
    # TODO
    def is_ready(self):
        """
            for live: e.g. exchange balances and positions are ready, how to know?
            for backtesting: backfilling is finished
        """
        pass
    
    def is_sub_strategy(self) -> bool:
        return bool(self._consumer)
    
    # TODO: add versioning, run_id etc.
    def to_dict(self):
        return {
            'class': self.__class__.__name__,
            'name': self.name,
            'config': self.config,
            'params': self.params,
            'accounts': [repr(account) for account in self.list_accounts()],
            'datas': [repr(data) for data in self.list_datas()],
            'strategies': [strategy.to_dict() for strategy in self.strategies.values()],
            'models': [model.to_dict() for model in self.models.values() if model.is_model()],
            'features': [model.to_dict() for model in self.models.values() if model.is_feature()],
            'indicators': [model.to_dict() for model in self.models.values() if model.is_indicator()],
            'strategy_signature': self._strategy_signature,
            'data_signatures': self._databoy._data_signatures,
        }
    
    def get_trading_venues(self) -> list[tTRADING_VENUE]:
        return list(self.accounts.keys())
    
    @overload
    def get_position(self, account: CryptoAccount, pdt: str) -> CryptoPosition | None: ...
    
    @overload
    def get_position(self, account: IBAccount, pdt: str) -> IBPosition | None: ...
    
    def get_position(self, account: BaseAccount, pdt: str) -> BasePosition | None:
        return self.positions[account].get(pdt, None)
    
    def list_positions(self) -> list[BasePosition]:
        return [position for pdt_to_positions in self.positions.values() for position in pdt_to_positions.values()]
    
    @overload
    def get_balance(self, account: CryptoAccount, ccy: str) -> CryptoBalance | None: ...
    
    @overload
    def get_balance(self, account: IBAccount, ccy: str) -> IBBalance | None: ...
    
    def get_balance(self, account: BaseAccount, ccy: str) -> BaseBalance | None:
        return self.balances[account].get(ccy, None)
    
    def list_balances(self) -> list[BaseBalance]:
        return [balance for ccy_to_balance in self.balances.values() for balance in ccy_to_balance.values()]
    
    @overload
    def get_account(self, trading_venue: tCRYPTO_EXCHANGE, name: str='') -> CryptoAccount: ...
        
    @overload
    def get_account(self, trading_venue: Literal['IB'], name: str='') -> IBAccount: ...
    
    def get_account(self, trading_venue: tTRADING_VENUE, name: str='') -> BaseAccount:
        trading_venue, name = trading_venue.upper(), name.upper()
        if not name:
            name = next(iter(self.accounts[trading_venue]))
            self.logger.warning(f"{trading_venue} account not specified, using first account '{name}'")
        return self.accounts[trading_venue][name]
    
    def list_accounts(self) -> list[BaseAccount]:
        return [account for accounts_per_trading_venue in self.accounts.values() for account in accounts_per_trading_venue.values()]
    
    @overload
    def add_account(
        self, 
        trading_venue: tCRYPTO_EXCHANGE, 
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
        trading_venue: tTRADING_VENUE,
        name: str='',
        initial_balances: dict[str, float] | None=None, 
        initial_positions: dict[BaseProduct, float] | None=None,
    ) -> SimulatedAccount: ...
    
    def add_account(self, trading_venue: tTRADING_VENUE, name: str='', **kwargs) -> BaseAccount:
        trading_venue, name = trading_venue.upper(), name.upper()
        broker: BaseBroker = self._engine.add_broker(trading_venue)
        if broker.name == Broker.CRYPTO:
            exch = trading_venue
            account =  broker.add_account(exch=exch, name=name or self.name, **kwargs)
        else:
            account = broker.add_account(name=name or self.name, **kwargs)
        if account.name not in self.accounts[trading_venue]:
            self.accounts[trading_venue][account.name] = account
            self.positions[account] = {}
            self.balances[account] = {}
            self.orders[account] = []
            # TODO make maxlen a variable in config
            self.trades[account] = deque(maxlen=10)
            self.logger.debug(f'added account {trading_venue=} {account.name=}')
        return account
    
    def add_data(
        self, 
        trading_venue: tTRADING_VENUE,
        product: str,
        exchange: str='',
        symbol: str='',
        data_source: tDATA_SOURCE | None=None,
        data_origin: str='',
        data_config: DataConfigDict | DataConfig | None=None,
        **product_specs
    ) -> list[TimeBasedData]:
        '''
        Args:
            exchange: useful for TradFi brokers, e.g. IB, to specify the exchange
            product: product basis, defined as {base_asset}_{quote_asset}_{product_type}, e.g. BTC_USDT_PERP
            product_specs: product specifications, e.g. expiration, strike_price etc.
        '''
        trading_venue: TradingVenue = TradingVenue[trading_venue.upper()]
        data_source = data_source or trading_venue.value
        if isinstance(data_config, dict):
            data_config['primary_resolution'] = self.resolution

        if not self.is_sub_strategy():
            broker: BaseBroker = self._engine.add_broker(trading_venue)
            if broker.name == Broker.CRYPTO:
                exch = trading_venue.value
                product: BaseProduct = broker.add_product(exch, product, symbol=symbol, **product_specs)
            elif broker.name == Broker.IB:
                product: BaseProduct = broker.add_product(product, exchange=exchange, symbol=symbol, **product_specs)
            else:
                raise NotImplementedError(f"Broker {broker.name} is not supported")
            datas: list[TimeBasedData] = self._databoy.add_data(product, data_config=data_config)
            for data in datas:
                self._add_data(data)
                if not data.is_resamplee():
                    # TODO: add channel to broker
                    broker.add_channel()
                broker._add_data_listener(self, data)
                if data.resolution == self.resolution:
                    mtstore = self._engine._store
                    mtstore.register_market_data(
                        consumer=self.name,
                        data_source=data_source,
                        data_origin=data_origin,
                        product=data.product,
                        resolution=self.resolution,
                        start_date=self.dataset_start,
                        end_date=self.dataset_end,
                    )
        else:
            datas: list[TimeBasedData] = self._add_data_to_consumer(
                trading_venue=trading_venue, 
                product=product, 
                symbol=symbol,
                data_source=data_source, 
                data_origin=data_origin, 
                data_config=data_config, 
                **product_specs
            )
        return datas
    
    # TODO, for website to remove data from a strategy
    # should check if broker still has listeners, if not, remove the data from broker
    # also need to consider products, need to remove product if no data is left
    # TODO: should call broker.remove_data()
    def remove_data(self, product: BaseProduct, resolution: str):
        if datas := self.get_data(product, resolution=resolution):
            datas = list(datas.values()) if not resolution else list(datas)
            broker = self.get_broker(product.bkr)
            for data in datas:
                del self._datas[data.product][repr(data.resolution)]
                timeframe = data.resolution.timeframe
                if timeframe.is_quote():
                    del self._orderbooks[data.product]
                if timeframe.is_tick():
                    del self._tradebooks[data.product]
                broker._remove_data_listener(self, data)
            if not self._datas[product]:
                del self._datas[product]
    
    def add_strategy(self, strategy: StrategyT, name: str='') -> StrategyT:
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
    
    def remove_strategy(self, name: str):
        if name in self.strategies:
            del self.strategies[name]
            self.logger.debug(f'removed sub-strategy {name}')
        else:
            self.logger.error(f'sub-strategy {name} cannot be found, failed to remove')
            
    def update_positions(self, position):
        self.positions[position.account][position.pdt] = position
        self.on_position(position.account, position)

    def update_balances(self, balance):
        self.balances[balance.account][balance.ccy] = balance
        self.on_balance(balance.account, balance)

    def update_orders(self, order, type_: Literal['submitted', 'opened', 'closed', 'amended']):
        if type_ in ['submitted', 'opened']:
            if order not in self.orders[order.account]:
                self.orders[order.account].append(order)
        elif type_ == 'closed':
            if order in self.orders[order.account]:
                self.orders[order.account].remove(order)
        self.on_order(order.account, order, type_)

    def update_trades(self, order, type_: Literal['partial', 'filled']):
        trade = order.trades[-1]
        self.trades[order.account].append(trade)
        self.on_trade(order.account, trade, type_)
    
    def _start_strategies(self):
        for strategy in self.strategies.values():
            strategy.start()
            
    # TODO
    def _next(self, data: BaseData):
        # NOTE: only sub-strategies have predict()
        # pred_y = self.predict(X)
        pass
    
    def _register_to_mtstore(self):
        mtstore = self._engine._store
        components = [*self.strategies.values(), *self.models.values(), *self.features.values(), *self.indicators.values()]
        for component in components:
            metadata = component.to_dict()
            if component.is_strategy():
                mtstore.register_strategy(self.name, component, metadata)
            elif component.is_model():
                mtstore.register_model(self.name, component, metadata)
            elif component.is_feature():
                mtstore.register_feature(self.name, component, metadata)
            elif component.is_indicator():
                mtstore.register_indicator(self.name, component, metadata)
    
    def start(self):
        if not self.is_running():
            self.add_datas()
            self._add_datas_from_consumer_if_none()
            self.add_strategies()
            self._start_strategies()
            self.add_models()
            self.add_features()
            self.add_indicators()
            self._start_models()
            self._prepare_df()
            if self._engine._use_ray:
                self._databoy.collect()
                # TODO: notice strategy manager it has started running
                pass
            self.on_start()

            self._register_to_mtstore()
            # TODO: self._store.materialize(), after on_start()?
            
            self._is_running = True
            self.logger.info(f"strategy '{self.name}' has started")
        else:
            self.logger.warning(f'strategy {self.name} has already started')
        
    def stop(self, reason: str=''):
        if self.is_running():
            self._is_running = False
            self.on_stop()
            if self._engine._use_ray:
                # TODO: notice strategy manager it has stopped running
                pass
            for strategy in self.strategies.values():
                strategy.stop(reason=reason)
            for model in self.models.values():
                model.stop()
            self.logger.info(f"strategy '{self.name}' has stopped, ({reason=})")
        else:
            self.logger.warning(f'strategy {self.name} has already stopped')

    def create_order(self, account: BaseAccount, product: BaseProduct, side: int, quantity: float, price: float | None=None, **kwargs):
        broker = self.get_broker(account.bkr)
        return broker.create_order(account, product, side, quantity, price=price, **kwargs)
    
    def place_orders(self, account: BaseAccount, product: BaseProduct, orders: list[BaseOrder]|BaseOrder):
        if not isinstance(orders, list):
            orders = [orders]
        broker = self.get_broker(account.bkr)
        broker.place_orders(account, product, orders)
        return orders

    def cancel_orders(self, account: BaseAccount, product: BaseProduct, orders: list[BaseOrder]|BaseOrder):
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
    Strategy Functions
    Users can customize these functions in their strategies.
    ************************************************
    '''
    def add_strategies(self):
        pass
    
    def on_quote(self, product, bids, asks, ts, **kwargs):
        raise NotImplementedError(f"Please define your own on_quote(product, bids, asks, ts, **kwargs) in your strategy '{self.name}'.")
    
    def on_tick(self, product, px, qty, ts, **kwargs):
        raise NotImplementedError(f"Please define your own on_tick(product, px, qty, ts, **kwargs) in your strategy '{self.name}'.")

    def on_bar(self, product, bar: Bar, ts, **kwargs):
        raise NotImplementedError(f"Please define your own on_bar(product, bar, ts, **kwargs) in your strategy '{self.name}'.")

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
    