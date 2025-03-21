from __future__ import annotations

import os
from collections import defaultdict, deque, OrderedDict
from abc import ABC

from typing import Literal, TYPE_CHECKING, overload
if TYPE_CHECKING:
    from pfeed.typing import tDATA_SOURCE
    from pfund.typing import tTRADING_VENUE, tBROKER, tCRYPTO_EXCHANGE
    from pfund.brokers.broker_base import BaseBroker
    from pfund.products.product_base import BaseProduct
    from pfund.positions.position_base import BasePosition
    from pfund.positions.position_crypto import CryptoPosition
    from pfund.positions.position_ib import IBPosition
    from pfund.balances.balance_base import BaseBalance
    from pfund.balances.balance_crypto import CryptoBalance
    from pfund.balances.balance_ib import IBBalance
    from pfund.typing import StrategyT, DataConfigDict, StorageConfigDict
    from pfund.datas.data_time_based import TimeBasedData
    from pfund.accounts.account_base import BaseAccount
    from pfund.accounts.account_crypto import CryptoAccount
    from pfund.accounts.account_ib import IBAccount
    from pfund.orders.order_base import BaseOrder
    from pfund.datas.data_base import BaseData
    from pfund.datas.data_bar import Bar
    from pfund.data_tools import data_tool_backtest
    from pfund.data_tools.data_tool_base import BaseDataTool
    from pfund.datas.data_config import DataConfig
    from pfund.datas.storage_config import StorageConfig

from pfund.strategies.strategy_meta import MetaStrategy
from pfund.engines import get_engine
from pfund.utils.envs import backtest
from pfund.mixins.trade_mixin import TradeMixin
from pfund.enums import Component, Broker


class BaseStrategy(TradeMixin, ABC, metaclass=MetaStrategy):    
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
        self.name = self.strat = self.get_default_name()
        self._engine = get_engine()
        self._data_tool: BaseDataTool = self._engine.DataTool()
        self.logger = None
        self._zmq = None
        self._is_parallel = False
        self._is_running = False
        self.datas = defaultdict(dict)  # {product: {'1m': data}}
        self._listeners = defaultdict(list)  # {data: model}
        self._consumers = []  # strategies that consume this strategy (sub-strategy)
        self.type = Component.strategy

        self.orderbooks = {}  # {product: data}
        self.tradebooks = {}  # {product: data}
        self.accounts = defaultdict(OrderedDict)  # {trading_venue: {acc1: account1, acc2: account2} }
        self.positions = {}  # {account: {pdt: position} }
        self.balances = {}  # {account: {ccy: balance}}
        # NOTE: includes submitted orders and opened orders
        self.orders = {}  # {account: [order, ...]}
        self.trades = {}  # {account: [trade, ...]}
        
        self.strategies = {}
        self.models = {}
        # NOTE: current strategy's signal is consumer's prediction
        self.predictions = {}  # {strat/mdl: pred_y}
        self._signals = {}  # {data: signal}, for strategy, signal is buy/sell/null
        self._last_signal_ts = {}  # {data: ts}
        self._signal_cols = []
        self._num_signal_cols = 0

        # TODO: move to mtflow
        self._data_signatures = []
        self._strategy_signature = (args, kwargs)
        
        self.params = {}
        self.load_params()
    
    # HACK: ideally no backtesting methods should be here, but it is required to improve IDE Intellisense
    @backtest
    def backtest(self, df: data_tool_backtest.BacktestDataFrame):
        raise Exception(f'Strategy "{self.name}" does not have a backtest() method, cannot run vectorized backtesting')
    
    def is_parallel(self):
        return self._is_parallel
    
    # TODO
    def is_ready(self):
        """
            for live: e.g. exchange balances and positions are ready, how to know?
            for backtesting: backfilling is finished
        """
        pass
    
    def is_sub_strategy(self):
        return bool(self._consumers)
    
    def to_dict(self):
        return {
            'class': self.__class__.__name__,
            'name': self.name,
            'config': self.config,
            'params': self.params,
            'accounts': [repr(account) for trading_venue in self.accounts for account in self.accounts[trading_venue].values()],
            'datas': [repr(data) for product in self.datas for data in self.datas[product].values()],
            'strategies': [strategy.to_dict() for strategy in self.strategies.values()],
            'models': [model.to_dict() for model in self.models.values()],
            'strategy_signature': self._strategy_signature,
            'data_signatures': self._data_signatures,
        }
    
    def get_trading_venues(self) -> list[tTRADING_VENUE]:
        return list(self.accounts.keys())
    
    def set_name(self, name: str):
        self.name = self.strat = name

    def set_parallel(self, is_parallel):
        self._is_parallel = is_parallel
  
    def pong(self):
        """Pongs back to Engine's ping to show that it is alive"""
        zmq_msg = (0, 0, (self.strat,))
        self._zmq.send(*zmq_msg, receiver='engine')

    def get_zmq(self):
        return self._zmq

    def start_zmq(self):
        from pfund.zeromq import ZeroMQ
        zmq_ports = self._engine.zmq_ports
        self._zmq = ZeroMQ(self.name)
        self._zmq.start(
            logger=self.logger,
            send_port=zmq_ports[self.name],
            recv_ports=[zmq_ports['engine']]
        )
        zmq_msg = (0, 1, (self.strat, os.getpid(),))
        self._zmq.send(*zmq_msg, receiver='engine')

    def stop_zmq(self):
        self._zmq.stop()
        self._zmq = None
    
    def add_broker(self, bkr: tBROKER) -> BaseBroker:
        return self._engine.add_broker(bkr)
    
    @overload
    def get_position(self, account: CryptoAccount, pdt: str) -> CryptoPosition | None: ...
    
    @overload
    def get_position(self, account: IBAccount, pdt: str) -> IBPosition | None: ...
    
    def get_position(self, account: BaseAccount, pdt: str) -> BasePosition | None:
        return self.positions[account].get(pdt, None)
    
    def get_positions(self) -> list[BasePosition]:
        return [position for pdt_to_positions in self.positions.values() for position in pdt_to_positions.values()]
    
    @overload
    def get_balance(self, account: CryptoAccount, ccy: str) -> CryptoBalance | None: ...
    
    @overload
    def get_balance(self, account: IBAccount, ccy: str) -> IBBalance | None: ...
    
    def get_balance(self, account: BaseAccount, ccy: str) -> BaseBalance | None:
        return self.balances[account].get(ccy, None)
    
    def get_balances(self) -> list[BaseBalance]:
        return [balance for ccy_to_balance in self.balances.values() for balance in ccy_to_balance.values()]
    
    @overload
    def get_account(self, trading_venue: tCRYPTO_EXCHANGE, acc: str='') -> CryptoAccount: ...
        
    @overload
    def get_account(self, trading_venue: Literal['IB'], acc: str='') -> IBAccount: ...
    
    def get_account(self, trading_venue: tTRADING_VENUE, acc: str='') -> BaseAccount:
        trading_venue, acc = trading_venue.upper(), acc.upper()
        if not acc:
            acc = next(iter(self.accounts[trading_venue]))
            self.logger.warning(f"{trading_venue} account not specified, using first account '{acc}'")
        return self.accounts[trading_venue][acc]
    
    def add_account(self, trading_venue: tTRADING_VENUE, acc: str='', **kwargs) -> BaseAccount:
        trading_venue, acc = trading_venue.upper(), acc.upper()
        bkr = self._derive_bkr_from_trading_venue(trading_venue)
        broker = self.add_broker(bkr)
        if bkr == 'CRYPTO':
            exch = trading_venue
            account =  broker.add_account(exch=exch, acc=acc, strat=self.strat, **kwargs)
        else:
            account = broker.add_account(acc=acc, strat=self.strat, **kwargs)
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
        resolution: str,
        data_source: tDATA_SOURCE | None=None,
        data_origin: str='',
        data_config: DataConfigDict | DataConfig | None=None,
        storage_config: StorageConfigDict | StorageConfig | None=None,
        **product_specs
    ) -> list[TimeBasedData]:
        '''
        Args:
            product: product basis, defined as {base_asset}_{quote_asset}_{product_type}, e.g. BTC_USDT_PERP
            product_specs: product specifications, e.g. expiration, strike_price etc.
        '''
        if not isinstance(data_config, DataConfig) or not isinstance(storage_config, StorageConfig):
            data_config, storage_config = self._parse_data_and_storage_config(trading_venue, resolution, data_source, data_config, storage_config)

        trading_venue, product = trading_venue.upper(), product.upper()
        bkr = self._derive_bkr_from_trading_venue(trading_venue)
        broker = self.add_broker(bkr)
        if bkr == Broker.CRYPTO:
            exch = trading_venue
            datas: list[TimeBasedData] = broker.add_data(exch, product, resolution, data_config=data_config, **product_specs)
        else:
            datas: list[TimeBasedData] = broker.add_data(product, resolution, data_config=data_config, **product_specs)

        for data in datas:
            if not data.is_resampler():
                continue
            resolution, product = data.resolution, data.product
            if resolution.is_quote():
                self.orderbooks[product] = data
            if resolution.is_tick():
                self.tradebooks[product] = data
            
            self.set_data(product, resolution, data)
            self._add_historical_data(data, data_source, data_origin, storage_config)
            if not self.is_sub_strategy():
                broker._add_listener(listener=self, listener_key=data, event_type='public')
            else:
                for consumer in self._consumers:
                    consumer.add_data(
                        trading_venue, 
                        product, 
                        resolution, 
                        data_source=data_source, 
                        data_origin=data_origin, 
                        data_config=data_config, 
                        storage_config=storage_config, 
                        **product_specs
                    )
                    consumer._add_listener(listener=self, listener_key=data)
                
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
                del self.datas[data.product][repr(data.resolution)]
                timeframe = data.resolution.timeframe
                if timeframe.is_quote():
                    del self.orderbooks[data.product]
                if timeframe.is_tick():
                    del self.tradebooks[data.product]
                broker._remove_listener(listener=self, listener_key=data, event_type='public')
            if not self.datas[product]:
                del self.datas[product]
    
    def add_strategy(self, strategy: StrategyT, name: str='', is_parallel=False) -> StrategyT:
        # TODO
        assert not is_parallel, 'Running strategy in parallel is not supported yet'
        assert isinstance(strategy, BaseStrategy), \
            f"strategy '{strategy.__class__.__name__}' is not an instance of BaseStrategy. Please create your strategy using 'class {strategy.__class__.__name__}(pf.Strategy)'"
        if name:
            strategy.set_name(name)
        strategy.set_parallel(is_parallel)
        strategy.create_logger()
        strat = strategy.name
        if strat in self.strategies:
            return self.strategies[strat]
        strategy._add_consumer(self)
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

    def _subscribe_to_private_channels(self):
        for broker in self.get_brokers():
            broker._add_listener(listener=self, listener_key=self.strat, event_type='private')
            
    # TODO
    def _next(self, data: BaseData):
        # NOTE: only sub-strategies have predict()
        # pred_y = self.predict(X)
        pass
    
    def start(self):
        if not self.is_running():
            self.add_datas()
            self._add_consumers_datas_if_no_data()
            self.add_strategies()
            self._start_strategies()
            self.add_models()
            self.add_features()
            self.add_indicators()
            self._start_models()
            self._prepare_df()
            self._subscribe_to_private_channels()
            if self.is_parallel():
                # TODO: notice strategy manager it has started running
                pass
                # self._zmq ...
            self.on_start()
            
            self._is_running = True
            self.logger.info(f"strategy '{self.name}' has started")
        else:
            self.logger.warning(f'strategy {self.name} has already started')
        
    def stop(self, reason: str=''):
        if self.is_running():
            self._is_running = False
            self.on_stop()
            if self.is_parallel():
                # TODO: notice strategy manager it has stopped running
                pass
                # self._zmq ...
            for broker in self.get_brokers():
                broker._remove_listener(listener=self, listener_key=self.strat, event_type='private')
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
    