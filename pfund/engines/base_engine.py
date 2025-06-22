from __future__ import annotations
from typing import TYPE_CHECKING, Literal, ClassVar
if TYPE_CHECKING:
    from mtflow.stores.mtstore import MTStore
    from mtflow.kernel import TradeKernel
    from mtflow.stores.trading_store import TradingStore
    from pfeed.typing import tDataTool
    from pfund.products.product_base import BaseProduct
    from pfund.accounts.account_base import BaseAccount
    from pfund.datas.data_time_based import TimeBasedData
    from pfund.typing import (
        StrategyT, 
        tEnvironment, 
        tBroker,
        tDatabase, 
        tTradingVenue,
        DataRangeDict, 
        ExternalListenersDict, 
        Component,
        ComponentName,
    )
    from pfund.brokers.broker_base import BaseBroker
    from pfund.strategies.strategy_base import BaseStrategy
    from pfund.engines.base_engine_settings import BaseEngineSettings

import logging
import datetime

from pfeed.enums import DataTool
from pfund import cprint
from pfund.engines.meta_engine import MetaEngine
from pfund.proxies.actor_proxy import ActorProxy
from pfund.enums import (
    Environment, 
    Broker, 
    RunMode, 
    Database, 
    CryptoExchange, 
    TradingVenue, 
)


ENV_COLORS = {
    # 'yellow': 'bold yellow on #ffffe0',
    # 'magenta': 'bold magenta on #fff0ff',
    # 'TRAIN': 'bold cyan on #d0ffff',
    Environment.BACKTEST: 'bold blue on #e0e0ff',
    Environment.SANDBOX: 'bold black on #f0f0f0',
    Environment.PAPER: 'bold red on #ffe0e0',
    Environment.LIVE: 'bold green on #e0ffe0',
}


class BaseEngine(metaclass=MetaEngine):
    '''
    _proxy: ZeroMQ xsub-xpub proxy for messaging from trading venues -> engine -> components
    _router: ZeroMQ router-pull for pulling messages from components (e.g. strategies/models) -> engine -> (routing to) trading venues
    _publisher: ZeroMQ publisher for broadcasting internal states to external apps
    '''
    
    _num: ClassVar[int] = 0
    _initialized: ClassVar[bool]
    _env: ClassVar[Environment]
    _run_mode: ClassVar[RunMode]
    _data_tool: ClassVar[DataTool]
    _data_start: ClassVar[datetime.date]
    _data_end: ClassVar[datetime.date]
    _database: ClassVar[tDatabase | None]
    _settings: ClassVar[BaseEngineSettings]
    _external_listeners: ClassVar[ExternalListenersDict | None]
    
    name: str
    _logger: logging.Logger
    _logging_config: dict
    _store: MTStore
    _kernel: TradeKernel
    brokers: dict[str, BaseBroker]
    strategies: dict[str, BaseStrategy]
    
    @classmethod
    def _next_engine_id(cls):
        cls._num += 1
        return str(cls._num)
    
    def _get_default_name(self):
        return f"{self.__class__.__name__}-{self._next_engine_id()}"
        
    def __init__(
        self, 
        *,
        env: tEnvironment, 
        name: str,
        data_tool: tDataTool,
        data_range: str | DataRangeDict | Literal['ytd'],
        database: tDatabase | None,
        settings: BaseEngineSettings,
        external_listeners: ExternalListenersDict | None,
    ):
        '''
        Args:
            external_listeners:
                If any of the keys is set to True, a websocket server will be started.
                This server listens to messages from ZeroMQ's PUB socket, and broadcasts them to connected external listeners.
                External listeners are programs typically created by mtflow and run outside the engine, such as:
                - Data Recorder
                - System Monitor
                - Profiler
                - Dashboards
                - Notebook apps
                These components subscribe to the broadcasted data in real-time.
            database:
                A database backend used for persisting data such as trades, orders, and internal states.
                If provided:
                    if `recorder` in `external_listeners` is set to True,
                        the DataRecorder will handle writing data to the database.
                    if `recorder` in `external_listeners` is set to False,
                        the engine itself will write to the database,
                        which will introduce latency and slow down engine's performance.
                If None, no data will be written.
        '''
        from mtflow.kernel import TradeKernel
        from mtflow.stores.mtstore import MTStore
        from pfeed.feeds.time_based_feed import TimeBasedFeed
        from pfund.external_listeners import ExternalListeners
        from pfund.utils.utils import derive_run_mode
        from pfund import get_config
        
        cls = self.__class__
        env = Environment[env.upper()]
        if not hasattr(cls, "_initialized"):
            cls._env = env
            cls._run_mode = derive_run_mode()
            cls._data_tool = DataTool[data_tool.lower()]
            cls._database = Database[database.upper()] if database else None
            cls._settings = settings
            if cls._run_mode == RunMode.WASM:
                assert not external_listeners, 'External listeners are not supported in WASM mode'
            cls._external_listeners = ExternalListeners(**(external_listeners or {}))
            if cls._external_listeners.recorder is False and cls._database is not None:
                cprint(
                    'WARNING: `database` is set but recorder is disabled in `external_listeners`, '
                    'data will be written to the database by the engine itself, which will introduce latency',
                    style='bold yellow'
                )
            is_data_range_dict = isinstance(data_range, dict)
            cls._data_start, cls._data_end = TimeBasedFeed._parse_date_range(
                start_date=data_range['start_date'] if is_data_range_dict else '',
                end_date=data_range.get('end_date', '') if is_data_range_dict else '',
                rollback_period=data_range if not is_data_range_dict else '',
            )
            cls._initialized = True
            cls.lock()  # Locks any future class modifications
            cprint(f"{env} Engine is running", style=ENV_COLORS[env])
        else:
            assert cls._env == env, f'Current environment is {cls._env}, cannot change to {env}'
            cprint("Engine already initialized — new inputs are ignored", style='bold yellow')

        
        # FIXME: do NOT allow LIVE env for now
        assert env != Environment.LIVE, f"{env=} is not allowed for now"
        

        self.name = name or self._get_default_name()
        
        config = get_config()
        config._load_env_file(self._env)
        self._logging_config = self._setup_logging()
        self._logger = logging.getLogger('pfund')

        self._store = MTStore(env=cls._env, data_tool=cls._data_tool)
        self._kernel = TradeKernel(
            engine_name=self.name,
            run_mode=cls._run_mode, 
            data_tool=cls._data_tool, 
            database=cls._database, 
            external_listeners=cls._external_listeners,
            settings=cls._settings,
        )

        self.trading_venues: list[TradingVenue] = []
        self.brokers: dict[Broker, BaseBroker] = {}
        self.strategies: dict[str, BaseStrategy | ActorProxy] = {}
    
    @property
    def env(self) -> Environment:
        return self._env
    
    @property
    def run_mode(self) -> RunMode:
        return self._run_mode
    
    @property
    def settings(self) -> BaseEngineSettings:
        return self._settings
    
    @property
    def data_start(self) -> datetime.date:
        return self._data_start
    
    @property
    def data_end(self) -> datetime.date:
        return self._data_end
    
    def _setup_logging(self) -> dict:
        from pfund import get_config
        from pfund._logging import setup_logging_config
        from pfund._logging.config import LoggingDictConfigurator
        config = get_config()
        log_path = f'{config.log_path}/{self._env}'
        user_logging_config = config.logging_config
        logging_config_file_path = config.logging_config_file_path
        logging_config = setup_logging_config(log_path, logging_config_file_path, user_logging_config=user_logging_config)
        # ≈ logging.config.dictConfig(logging_config) with a custom configurator
        logging_configurator = LoggingDictConfigurator(logging_config)
        logging_configurator.configure()
        return logging_configurator._pfund_config
    
    def run(self):
        self._store._freeze()
        local_strategies = []
        for strat, strategy in self.strategies.items():
            trading_store: TradingStore = self._store.get_trading_store(strat)
            strategy._set_trading_store(trading_store)
            # NOTE: if strategy is remote, trading store will be copied to it and
            # the one inside the engine is just a reference and should be frozen
            if strategy.is_remote():
                trading_store.freeze()
            else:
                local_strategies.append(strategy)
            # TODO:
            # strategy.start()
        if local_strategies:
            data = self._kernel.messenger.recv()
            for strategy in local_strategies:
                strategy.databoy._collect(data)
        
    # FIXME
    def is_running(self) -> bool:
        return self._kernel._is_running
    
    def add_strategy(
        self, 
        strategy: StrategyT, 
        resolution: str, 
        name: str='', 
        ray_actor_options: dict | None=None,
        **ray_kwargs
    ) -> StrategyT | ActorProxy:
        '''
        Args:
            ray_actor_options:
                Options for Ray actor.
                will be passed to ray actor like this: Actor.options(**ray_options).remote(**ray_kwargs)
        '''
        from pfund.strategies.strategy_base import BaseStrategy
        from pfund.utils.utils import derive_run_mode
        
        Strategy = strategy.__class__
        StrategyName = Strategy.__name__
        assert isinstance(strategy, BaseStrategy), \
            f"strategy '{StrategyName}' is not an instance of BaseStrategy. Please create your strategy using 'class {StrategyName}(pf.Strategy)'"
        
        strat = name or strategy.name
        if strat in self.strategies:
            raise ValueError(f"{strat} already exists")

        run_mode: RunMode = derive_run_mode(ray_kwargs)
        if is_remote := (run_mode == RunMode.REMOTE):
            strategy = ActorProxy(strategy, name=name, ray_actor_options=ray_actor_options, **ray_kwargs)
            strategy._set_proxy(strategy)
        strategy._set_name(strat)
        strategy._set_run_mode(run_mode)
        strategy._set_resolution(resolution)
        strategy._setup_logging(self._logging_config)
        strategy._set_engine(engine=None if is_remote else self, engine_settings=self.settings)

        self.strategies[strat] = strategy
        self._logger.debug(f"added '{strat}'")
        return strategy
    
    def get_strategy(self, name: str) -> BaseStrategy | ActorProxy:
        return self.strategies[name]
    
    def _create_broker(self, bkr: tBroker) -> BaseBroker:
        if self._env in [Environment.BACKTEST, Environment.SANDBOX]:
            from pfund.brokers.broker_simulated import SimulatedBrokerFactory
            SimulatedBroker = SimulatedBrokerFactory(bkr)
            broker = SimulatedBroker(self._env)
        else:
            BrokerClass = Broker[bkr.upper()].broker_class
            broker = BrokerClass(self._env)
        return broker

    def _add_broker(self, trading_venue: tTradingVenue) -> BaseBroker:
        trading_venue = TradingVenue[trading_venue.upper()]
        if trading_venue not in self.trading_venues:
            self.trading_venues.append(trading_venue)
        if trading_venue in CryptoExchange.__members__:
            bkr = Broker.CRYPTO
        else:
            bkr = Broker[trading_venue]
        if bkr not in self.brokers:
            broker = self._create_broker(bkr)
            self.brokers[bkr] = broker
            self._logger.debug(f'added broker {bkr}')
        return self.brokers[bkr]
    
    def get_broker(self, bkr: tBroker) -> BaseBroker:
        return self.brokers[bkr.upper()]
    
    def _register_product(
        self,
        trading_venue: tTradingVenue, 
        basis: str,
        exchange: str='', 
        symbol: str='', 
        name: str='',
        **specs
    ) -> BaseProduct:
        broker: BaseBroker = self._add_broker(trading_venue)
        if broker.name == Broker.CRYPTO:
            exch = trading_venue
            product: BaseProduct = broker.add_product(exch, basis, name=name, **specs)
        elif broker.name == Broker.IB:
            product: BaseProduct = broker.add_product(basis, exchange=exchange, symbol=symbol, name=name, **specs)
        else:
            raise NotImplementedError(f"Broker {broker.name} is not supported")
        return product
    
    def _register_account(self, trading_venue: tTradingVenue, name: str='', **kwargs) -> BaseAccount:
        from pfund.accounts.account_simulated import SimulatedAccount
        if 'initial_balances' in kwargs or 'initial_positions' in kwargs:
            assert self._env == Environment.SANDBOX, \
                f"initial balances and positions can only be set in {Environment.SANDBOX} environment"
        broker: BaseBroker = self._add_broker(trading_venue)
        if broker.name == Broker.CRYPTO:
            exch = trading_venue
            account =  broker.add_account(exch=exch, name=name or self.name, **kwargs)
        elif broker.name == Broker.IB:
            account = broker.add_account(name=name or self.name, **kwargs)
        else:
            raise NotImplementedError(f"Broker {broker.name} is not supported")
        if self._env == Environment.SANDBOX:
            assert isinstance(account, SimulatedAccount)
        return account
    
    def _register_component(self, consumer_name: ComponentName | None, component_name: ComponentName, component_metadata: dict):
        if component_metadata['run_mode'] == RunMode.REMOTE:
            self._kernel.add_ray_actor(component_name)
        self._store.register_component(consumer_name, component_name, component_metadata)
    
    def _register_market_data(self, component: Component, data: TimeBasedData):
        product = data.product
        if not data.is_resamplee():
            broker: BaseBroker = self.get_broker(product.bkr)
            broker._add_data_channel(data)
        self._store.register_market_data(
            consumer=component.name,
            data_source=data.data_source,
            data_origin=data.data_origin,
            product=product,
            resolution=data.resolution,
            start_date=self._data_start,
            end_date=self._data_end,
        )
