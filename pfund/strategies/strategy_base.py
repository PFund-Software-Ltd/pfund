# NOTE: need this to make TYPE_CHECKING work to avoid the circular import issue
from __future__ import annotations

import os
import sys
import time
import importlib
import datetime
from pathlib import Path
from collections import defaultdict, deque
from abc import ABC

from typing import Literal, TYPE_CHECKING
if TYPE_CHECKING:
    from pfund.brokers.broker_base import BaseBroker
    from pfund.exchanges.exchange_base import BaseExchange
    from pfund.datas.data_bar import Bar
    from pfund.types.core import tStrategy, tModel, tIndicator, tFeature, tProduct
    from pfund.products.product_base import BaseProduct
    from pfund.accounts.account_base import BaseAccount
    from pfund.orders.order_base import BaseOrder
    from pfund.models.model_base import BaseModel

from pfund.datas.resolution import Resolution
from pfund.datas import BaseData, BarData, TickData, QuoteData
from pfund.zeromq import ZeroMQ
from pfund.risk_monitor import RiskMonitor
from pfund.const.common import SUPPORTED_CRYPTO_EXCHANGES
from pfund.strategies.strategy_meta import MetaStrategy
from pfund.utils.utils import convert_to_uppercases, get_engine_class, load_yaml_file, convert_ts_to_dt
from pfund.plogging import create_dynamic_logger


class BaseStrategy(ABC, metaclass=MetaStrategy):
    REQUIRED_FUNCTIONS = ['on_quote', 'on_tick', 'on_bar']
    
    _file_path: Path | None = None  # Get the file path where the strategy was defined
    config = {}

    @classmethod
    def load_config(cls, config: dict | None=None):
        if config:
            cls.config = config
        elif cls._file_path:
            for file_name in ['config.yml', 'config.yaml']:
                if config := load_yaml_file(cls._file_path.parent / file_name):
                    cls.config = config
                    break
    
    def load_params(self, params: dict | None=None):
        if params:
            self.params = params
        elif self._file_path:
            for file_name in ['params.yml', 'params.yaml']:
                if params := load_yaml_file(self._file_path.parent / file_name):
                    self.params = params
                    break
    
    def __new__(cls, *args, **kwargs):
        if not cls._file_path:
            module = sys.modules[cls.__module__]
            if strategy_file_path := getattr(module, '__file__', None):
                cls._file_path = Path(strategy_file_path)
                cls.load_config()
        return super().__new__(cls)
    
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
        self.name = self.strat = self.__class__.__name__
        self.Engine = get_engine_class()
        self.engine = self.Engine()
        data_tool: str = self.Engine.data_tool
        DataTool = getattr(importlib.import_module(f'pfund.data_tools.data_tool_{data_tool}'), f'{data_tool.capitalize()}DataTool')
        self._data_tool = DataTool()
        self.logger = None
        self._zmq = None
        self._is_parallel = False
        self._is_running = False
        self.brokers = {}
        self.exchanges = {}
        self.products = defaultdict(dict)  # {trading_venue: {pdt1: product1, pdt2: product2, exch1_pdt3: product, exch2_pdt3: product} }
        self.accounts = defaultdict(dict)  # {trading_venue: {acc1: account1, acc2: account2} }
        self.datas = defaultdict(dict)  # {product: {'1m': data}}
        self._listeners = defaultdict(list)  # {data: model}
        self._consumers = []  # strategies that consume this strategy (sub-strategy)
        self.type = 'strategy'
        self.orderbooks = {}  # {product: data}
        self.tradebooks = {}  # {product: data}
        self.positions = {}  # {account: {pdt: position} }
        self.balances = {}  # {account: {ccy: balance}}
        # NOTE: includes submitted orders and opened orders
        self.orders = {}  # {account: [order, ...]}
        self.trades = {}  # {account: [trade, ...]}
        self.models = {}
        self.strategies = {}
        self.predictions = {}
        self.data = None  # last data
        self.signal = None  # output signal df where a signal is buy/sell/null
        
        # TODO
        self.portfolio = None
        self.universe = None
        self.investment_profile = None
        self.risk_monitor = self.rm = RiskMonitor()
        
        self.params = {}
        self.load_params()
    
    @property
    def tname(self):
        '''type + name, e.g. strategy SMA'''
        return f"{self.type} '{self.name}'"
    
    @property
    def df(self):
        return self._data_tool.df
    
    def get_df(self, copy=True):
        return self._data_tool.get_df(copy=copy)
   
    def get_data_tool(self):
        return self._data_tool
    
    @staticmethod
    def dt(ts: float):
        return convert_ts_to_dt(ts)
    
    def create_logger(self):
        self.logger = create_dynamic_logger(self.name, 'strategy')

    def is_parallel(self):
        return self._is_parallel
    
    def is_running(self):
        return self._is_running
    
    # TODO
    def is_ready(self):
        """
            for live: e.g. exchange balances and positions are ready, how to know?
            for backtesting: backfilling is finished
        """
        pass
    
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
        }
    
    def output_df_to_parquet(self, file_path: str):
        self._data_tool.output_df_to_parquet(self.df, file_path)
    
    def set_name(self, name: str):
        self.name = self.strat = name

    def set_parallel(self, is_parallel):
        self._is_parallel = is_parallel
    
    def add_consumer(self, consumer: BaseStrategy):
        if consumer not in self._consumers:
            self._consumers.append(consumer)        
    
    def add_listener(self, listener: BaseStrategy | BaseModel, listener_key: BaseData):
        if listener not in self._listeners[listener_key]:
            self._listeners[listener_key].append(listener)
    
    def remove_listener(self, listener: BaseStrategy | BaseModel, listener_key: BaseData):
        if listener in self._listeners[listener_key]:
            self._listeners[listener_key].remove(listener)
    
    @staticmethod
    def now():
        return datetime.datetime.now(tz=datetime.timezone.utc)
    
    def pong(self):
        """Pongs back to Engine's ping to show that it is alive"""
        zmq_msg = (0, 0, (self.strat,))
        self._zmq.send(*zmq_msg, receiver='engine')

    def get_zmq(self):
        return self._zmq

    def start_zmq(self):
        zmq_ports = self.engine.zmq_ports
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
    
    def get_delay(self, ts):
        return time.time() - ts
    
    def add_strategy(self, strategy: tStrategy, name: str='', is_parallel=False) -> tStrategy:
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
            raise Exception(f"sub-strategy '{strat}' already exists in strategy '{self.name}'")
        strategy.add_consumer(self)
        self.strategies[strat] = strategy
        self.logger.debug(f"added sub-strategy '{strat}'")
        return strategy
    
    def get_model(self, name: str) -> BaseModel:
        return self.models[name]
    
    def add_model(
        self, 
        model: tModel, 
        name: str='',
        min_data: int=1,
        max_data: None | int=None
    ) -> tModel:
        Model = model.get_model_type_of_ml_model()
        assert isinstance(model, Model), \
            f"{model.type} '{model.__class__.__name__}' is not an instance of {Model.__name__}. Please create your {model.type} using 'class {model.__class__.__name__}({Model.__name__})'"
        if name:
            model.set_name(name)
        model.set_min_data(min_data)
        model.set_max_data(max_data)
        model.create_logger()
        mdl = model.name
        if mdl in self.models:
            raise Exception(f"{model.tname} already exists in {self.tname}")
        model.add_consumer(self)
        self.models[mdl] = model
        self.logger.debug(f"added {model.tname}")
        return model
    
    def add_feature(
        self, 
        feature: tFeature, 
        name: str='',
        min_data: int=1,
        max_data: None | int=None
    ) -> tFeature:
        return self.add_model(feature, name=name, min_data=min_data, max_data=max_data)
    
    def add_indicator(
        self, 
        indicator: tIndicator, 
        name: str='',
        min_data: int=1,
        max_data: None | int=None
    ) -> tIndicator:
        return self.add_model(indicator, name=name, min_data=min_data, max_data=max_data)
    
    # TODO
    def add_custom_data(self):
        pass
    
    def get_datas(self) -> list[BaseData]:
        datas = []
        for product in self.datas:
            datas.extend(list(self.datas[product].values()))
        return datas
    
    def set_data(self, product: BaseProduct, resolution: str | Resolution, data: BaseData):
        if isinstance(resolution, Resolution):
            resolution = repr(resolution)
        self.datas[product][resolution] = data

    def get_data(self, product: BaseProduct, resolution: str | None=None):
        return self.datas[product] if not resolution else self.datas[product][resolution]

    def add_data(self, trading_venue, base_currency, quote_currency, ptype, *args, **kwargs) -> list[BaseData]:
        from pfund.managers.data_manager import get_resolutions_from_kwargs
        assert not ('resolution' in kwargs and 'resolutions' in kwargs), "Please use either 'resolution' or 'resolutions', not both"
        trading_venue, base_currency, quote_currency, ptype = convert_to_uppercases(trading_venue, base_currency, quote_currency, ptype)
        bkr = 'CRYPTO' if trading_venue in SUPPORTED_CRYPTO_EXCHANGES else trading_venue
        broker = self.add_broker(bkr) if bkr not in self.brokers else self.get_broker(bkr)
        
        if bkr == 'CRYPTO':
            exch = trading_venue
            datas = broker.add_data(exch, base_currency, quote_currency, ptype, *args, **kwargs)
        else:
            datas = broker.add_data(base_currency, quote_currency, ptype, *args, **kwargs)
   
        for data in datas:
            if data.is_time_based():
                # do not listen to data thats only used for resampling
                if data.resolution not in get_resolutions_from_kwargs(kwargs):
                    continue
            product = self.add_product(data.product)
            resolution = data.resolution
            if resolution.is_quote():
                self.orderbooks[product] = data
            if resolution.is_tick():
                self.tradebooks[product] = data
            
            is_sub_strategy = bool(self._consumers)
            if not is_sub_strategy:
                self.set_data(product, resolution, data)
                broker.add_listener(listener=self, listener_key=data, event_type='public')

        for consumer in self._consumers:
            self._add_consumer_datas(consumer, trading_venue, base_currency, quote_currency, ptype, *args, **kwargs)

        return datas
    
    def _add_consumer_datas(self, consumer: BaseStrategy | BaseModel, *args, use_consumer_data=False, **kwargs) -> list[BaseData]:
        if not use_consumer_data:
            consumer_datas = consumer.add_data(*args, **kwargs)
        else:
            consumer_datas = consumer.get_datas()
        for data in consumer_datas:
            self.set_data(data.product, data.resolution, data)
            consumer.add_listener(listener=self, listener_key=data)
        return consumer_datas
    
    def _add_datas_if_not_exist(self):
        if self.datas:
            return
        self.logger.warning(f"No data for {self.tname}, adding datas from consumers {[consumer.tname for consumer in self._consumers]}")
        for consumer in self._consumers:
            self._add_consumer_datas(consumer, use_consumer_data=True)

    # TODO, for website to remove data from a strategy
    def remove_data(self, product: BaseProduct, resolution: str | None=None):
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
                broker.remove_listener(listener=self, listener_key=data, event_type='public')
            if not self.datas[product]:
                del self.datas[product]
                self.remove_product(product)

    def get_account(self, trading_venue: str, acc: str=''):
        if acc:
            return self.accounts[trading_venue.upper()][acc.upper()]
        else:
            assert len(self.accounts[trading_venue.upper()]) == 1, f'{trading_venue} has more than one account, please specify the account name'
            return getattr(self, f'{trading_venue.lower()}_account')
    
    def add_account(self, trading_venue: str, acc: str='', **kwargs) -> BaseAccount:
        trading_venue, acc = trading_venue.upper(), acc.upper()
        bkr = 'CRYPTO' if trading_venue in SUPPORTED_CRYPTO_EXCHANGES else trading_venue
        broker = self.add_broker(bkr) if bkr not in self.brokers else self.get_broker(bkr)
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
        else:
            raise Exception(f'Strategy {self.name} already has an account called {acc} set up for {trading_venue=}')
        return account
    
    def get_product(self, trading_venue: str, pdt: str='', exch: str='') -> BaseProduct:
        if pdt:
            if not exch:
                return self.products[trading_venue.upper()][pdt.upper()]
            else:
                # same product, different exchanges, needs a different key
                return self.products[trading_venue.upper()][exch.upper()+'_'+pdt.upper()]
        else:
            assert len(self.products[trading_venue.upper()]) == 1, f'{trading_venue} has more than one product, please specify the product name'
            return getattr(self, f'{trading_venue.lower()}_product')

    def add_product(self, product: tProduct, pdt_key='') -> tProduct:
        trading_venue = product.exch if product.bkr == 'CRYPTO' else product.bkr
        pdt_key = pdt_key or product.pdt
        if pdt_key not in self.products[trading_venue]:
            self.products[trading_venue][pdt_key] = product
            if product.is_crypto() and product.exch not in self.exchanges:
                self.add_exchange(product.exch)
            self.logger.debug(f'added product {product}')
            return product
        else:
            exist_product = self.products[trading_venue][pdt_key]
            # change the pdt_key if same product, different exchanges
            if product.exch != exist_product.exch:
                self.remove_product(exist_product)
                self.add_product(exist_product, pdt_key=exist_product.exch+'_'+product.pdt)
                return self.add_product(product, pdt_key=product.exch+'_'+product.pdt)
            return exist_product
    
    def remove_product(self, product: BaseProduct):
        trading_venue = product.exch if product.bkr == 'CRYPTO' else product.bkr
        if product.pdt in self.products[trading_venue]:
            del self.products[trading_venue][product.pdt]
            if product.is_crypto() and not self.products[trading_venue]:
                del self.products[trading_venue]
                self.remove_exchange(product.exch)
            self.logger.debug(f'removed product {product}')

    def get_exchange(self, exch) -> BaseExchange: 
        return self.exchanges[exch.upper()]
    
    def add_exchange(self, exch: str):
        broker = self.get_broker('CRYPTO')
        self.exchanges[exch] = broker.get_exchange(exch)
        self.logger.debug(f'added exchange {exch}')

    def remove_exchange(self, exch: str):
        del self.exchanges[exch.upper()]
        self.logger.debug(f'removed exchange {exch}')

    def get_broker(self, bkr: str) -> BaseBroker:
        return self.brokers[bkr.upper()]

    def add_broker(self, bkr: str) -> BaseBroker:
        broker = self.engine.add_broker(bkr)
        self.brokers[bkr] = broker
        return broker

    def update_quote(self, data: QuoteData, **kwargs):
        product, bids, asks, ts = data.product, data.bids, data.asks, data.ts
        self.data = data
        for listener in self._listeners[data]:
            listener.update_quote(data, **kwargs)
            self.update_predictions(data, listener)
        self._append_to_df(**kwargs)
        self.on_quote(product, bids, asks, ts, **kwargs)

    def update_tick(self, data: TickData, **kwargs):
        product, px, qty, ts = data.product, data.px, data.qty, data.ts
        self.data = data
        for listener in self._listeners[data]:
            listener.update_tick(data, **kwargs)
            self.update_predictions(data, listener)
        self._append_to_df(**kwargs)
        self.on_tick(product, px, qty, ts, **kwargs)
    
    def update_bar(self, data: BarData, **kwargs):
        product, bar, ts = data.product, data.bar, data.bar.end_ts
        self.data = data
        for listener in self._listeners[data]:
            # NOTE: listener could be a strategy or a model
            listener.update_bar(data, **kwargs)
            self.update_predictions(data, listener)
        self._append_to_df(**kwargs)
        self.on_bar(product, bar, ts, **kwargs)

    def update_predictions(self, data: BaseData, listener: BaseStrategy | BaseModel):
        pred_y = listener.next(data)
        self.predictions[listener.name] = pred_y
        
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

    def _start_models(self):
        for model in self.models.values():
            model.start()
    
    def _subscribe_to_private_channels(self):
        for broker in self.brokers.values():
            broker.add_listener(listener=self, listener_key=self.strat, event_type='private')
    
    def _prepare_df(self):
        return self._data_tool.prepare_df(ts_col_type='timestamp')
        
    def _append_to_df(self, **kwargs):
        return self._data_tool.append_to_df(self.data, self.predictions, **kwargs)
    
    def start(self):
        if not self.is_running():
            self.add_datas()
            self._add_datas_if_not_exist()
            self.add_strategies()
            self._start_strategies()
            self.add_models()
            self._start_models()
            self._prepare_df()
            self._subscribe_to_private_channels()
            self._set_aliases()
            if self.is_parallel():
                # TODO: notice strategy manager it has started running
                pass
                # self._zmq ...
            self.on_start()
            self._is_running = True
        else:
            self.logger.warning(f'strategy {self.name} has already started')
        
    def stop(self, reason=''):
        if self.is_running():
            self._is_running = False
            self.on_stop()
            if self.is_parallel():
                # TODO: notice strategy manager it has stopped running
                pass
                # self._zmq ...
            for broker in self.brokers.values():
                broker.remove_listener(listener=self, listener_key=self.strat, event_type='private')
            for strategy in self.strategies.values():
                strategy.stop()
            for model in self.models.values():
                model.stop()
        else:
            self.logger.warning(f'strategy {self.name} has already stopped')

    def create_order(self, account: BaseAccount, product: BaseProduct, side: int, quantity: float, price=None, **kwargs):
        broker = self.get_broker(account.bkr)
        return broker.create_order(account, product, side, quantity, price=price, **kwargs)
    
    def place_orders(self, account: BaseAccount, product: BaseProduct, orders: list[BaseOrder]|BaseOrder):
        if type(orders) is not list:
            orders = [orders]
        broker = self.get_broker(account.bkr)
        broker.place_orders(account, product, orders)
        return orders

    def cancel_orders(self, account: BaseAccount, product: BaseProduct, orders: list[BaseOrder]|BaseOrder):
        if type(orders) is not list:
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
    def add_datas(self):
        pass
    
    def add_strategies(self):
        pass
    
    def add_models(self):
        pass
    
    def on_start(self):
        pass
    
    def on_stop(self):
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
    def _set_aliases(self):
        for name, products_or_accounts in [('product', self.products), ('account', self.accounts)]:
            for tv in products_or_accounts:
                num_product_or_account = len(products_or_accounts[tv])
                # e.g. if Interactive Brokers has only one account, 
                # create attributes `self.ib_account` `self.ib_positions` for quicker access
                if num_product_or_account == 1:
                    product_or_account = list(products_or_accounts[tv].values())[0]
                    tv = tv.lower()
                    setattr(self, f'{tv}_{name}', product_or_account)
                    if name == 'account':
                        account = product_or_account
                        setattr(self, f'{tv}_positions', self.positions[account])
                        setattr(self, f'{tv}_balances', self.balances[account])
                        setattr(self, f'{tv}_orders', self.orders[account])
                        setattr(self, f'{tv}_trades', self.trades[account])
                    # if also only has one trading_venue, set aliases self.account and self.product
                    num_tvs = len(products_or_accounts)
                    if num_tvs == 1:
                        setattr(self, f'{name}', product_or_account)
    
    def get_second_bar(self, product: BaseProduct, period: int):
        return self.get_data(product, resolution=f'{period}_SECOND')
    
    def get_minute_bar(self, product: BaseProduct, period: int):
        return self.get_data(product, resolution=f'{period}_MINUTE')
    
    def get_hour_bar(self, product: BaseProduct, period: int):
        return self.get_data(product, resolution=f'{period}_HOUR')
    
    def get_day_bar(self, product: BaseProduct, period: int):
        return self.get_data(product, resolution=f'{period}_DAY')
    
    def get_week_bar(self, product: BaseProduct, period: int):
        return self.get_data(product, resolution=f'{period}_WEEK')
    
    def get_month_bar(self, product: BaseProduct, period: int):
        return self.get_data(product, resolution=f'{period}_MONTH')
    
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
    