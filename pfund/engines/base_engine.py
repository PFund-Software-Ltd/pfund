from __future__ import annotations
from typing import TYPE_CHECKING, Literal, ClassVar
if TYPE_CHECKING:
    from ray.actor import ActorHandle
    from mtflow.stores.mtstore import MTStore
    from mtflow.kernel import TradeKernel
    from pfeed.typing import tDATA_TOOL
    from pfund.typing import StrategyT, tENVIRONMENT, tBROKER, DataRangeDict, tDATABASE, tTRADING_VENUE
    from pfund.typing import ExternalListenersDict
    from pfund.brokers.broker_base import BaseBroker
    from pfund.strategies.strategy_base import BaseStrategy
    from pfund.models.model_base import BaseModel
    from pfund.engines.trade_engine_settings import TradeEngineSettings
    from pfund.engines.backtest_engine_settings import BacktestEngineSettings

import logging
import datetime

from pfeed.enums import DataTool
from pfund import cprint
from pfund.engines.meta_engine import MetaEngine
from pfund.enums import Environment, Broker, RunMode, Database, CryptoExchange, TradingVenue
from pfund.external_listeners import ExternalListeners


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
    _num: ClassVar[int] = 0
    _initialized: ClassVar[bool]
    _env: ClassVar[Environment]
    _run_mode: ClassVar[RunMode]
    _data_tool: ClassVar[DataTool]
    _dataset_start: ClassVar[datetime.date]
    _dataset_end: ClassVar[datetime.date]
    _database: ClassVar[tDATABASE | None]
    _settings: ClassVar[TradeEngineSettings | BacktestEngineSettings]
    _external_listeners: ClassVar[ExternalListenersDict | None]
    
    name: str
    _logger: logging.Logger
    _store: MTStore
    _kernel: TradeKernel | None
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
        env: tENVIRONMENT, 
        name: str,
        data_tool: tDATA_TOOL,
        data_range: str | DataRangeDict | Literal['ytd'],
        use_ray: bool,
        database: tDATABASE | None,
        settings: TradeEngineSettings | BacktestEngineSettings,
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
                        which will introduce latency and slow down engine performance.
                If None, no data will be written.
        '''
        from pfeed.feeds.time_based_feed import TimeBasedFeed
        from mtflow.utils.utils import is_wasm
        from mtflow.kernel import TradeKernel
        from mtflow.stores.mtstore import MTStore
        
        cls = self.__class__
        env = Environment[env.upper()]
        if not hasattr(cls, "_initialized"):
            cls._env = env
            if is_wasm():
                cls._run_mode = RunMode.WASM
                assert not use_ray, 'Ray is not supported in WASM mode'
            else:
                cls._run_mode = RunMode.REMOTE if use_ray else RunMode.LOCAL
            cls._data_tool = DataTool[data_tool.lower()]
            cls._database = Database[database.upper()] if database else None
            cls._settings = settings
            cls._external_listeners = ExternalListeners(**(external_listeners or {}))
            if cls._external_listeners.recorder is False and cls._database is not None:
                cprint(
                    'WARNING: `database` is set but recorder is disabled in `external_listeners`, '
                    'data will be written to the database by the engine itself, which will introduce latency',
                    style='bold yellow'
                )
            is_data_range_dict = isinstance(data_range, dict)
            cls._dataset_start, cls._dataset_end = TimeBasedFeed._parse_date_range(
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
        self._setup_logging()
        self._logger = logging.getLogger('pfund')
        self._store = MTStore(env=cls._env, data_tool=cls._data_tool)
        self._kernel = TradeKernel(
            mode=cls._run_mode,
            database=cls._database,
            external_listeners=cls._external_listeners,
            zmq_urls=cls._settings.zmq_urls,
            zmq_ports=cls._settings.zmq_ports,
        )
        self.brokers: dict[Broker, BaseBroker] = {}
        self.strategies: dict[str, BaseStrategy] = {}
    
    @property
    def env(self) -> Environment:
        return self._env
    
    @property
    def run_mode(self) -> RunMode:
        return self._run_mode
    
    @property
    def settings(self) -> TradeEngineSettings | BacktestEngineSettings:
        return self._settings
    
    @property
    def dataset_start(self) -> datetime.date:
        return self._dataset_start
    
    @property
    def dataset_end(self) -> datetime.date:
        return self._dataset_end
    
    def _setup_logging(self):
        from pfund._logging import setup_loggers
        from pfund import get_config
        config = get_config()
        log_path = f'{config.log_path}/{self._env.lower()}'
        logging_config_file_path = config.logging_config_file_path
        logging_configurator  = setup_loggers(log_path, logging_config_file_path, user_logging_config=config.logging_config)
        config.set_logging_configurator(logging_configurator)
    
    @staticmethod
    def _is_ray_actor(value) -> bool:
        from ray.actor import ActorHandle
        return isinstance(value, ActorHandle)
    
    def _setup_ray_actors(self):
        def _add_ray_actor(component: BaseStrategy | BaseModel):
            databoy = component._databoy
            databoy._setup_zmq()
            self._kernel.add_ray_actor(component)
            for subcomponent in component.components:
                if self._is_ray_actor(subcomponent):
                    _add_ray_actor(subcomponent)
        for strategy in self.strategies.values():
            _add_ray_actor(strategy)

    def run(self):
        self._store.freeze()
        if self._kernel is not None and self._run_mode == RunMode.REMOTE:
            self._setup_ray_actors()
    
    # FIXME
    def is_running(self) -> bool:
        return self._kernel._is_running
    
    def add_strategy(self, strategy: StrategyT | ActorHandle, resolution: str, name: str='') -> StrategyT | ActorHandle:
        if self._is_ray_actor(strategy):
            import ray
            assert self._run_mode == RunMode.REMOTE, \
                f'Ray actors can only be added in remote mode, but current run mode is {self._run_mode}, please set use_ray=True'
            StrategyClass = ray.get(strategy._get_pfund_strategy_class.remote())
            assert issubclass(StrategyClass, BaseStrategy), \
                f"strategy '{StrategyClass.__name__}' is not an instance of BaseStrategy. Please create your strategy using 'class {StrategyClass.__name__}(BaseStrategy)'"
        else:
            assert isinstance(strategy, BaseStrategy), \
                f"strategy '{strategy.__class__.__name__}' is not an instance of BaseStrategy. Please create your strategy using 'class {strategy.__class__.__name__}(BaseStrategy)'"
        if name:
            strategy._set_name(name)
        strategy._set_resolution(resolution)
        strat = strategy.name
        if strat in self.strategies:
            raise ValueError(f"{strat} already exists")
        else:
            strategy._set_engine(self)
            trading_store = self._store.add_trading_store(strat)
            strategy._set_trading_store(trading_store)
            strategy._create_logger()
            self.strategies[strat] = strategy
            self._logger.debug(f"added '{strat}'")
            return strategy
    
    def remove_strategy(self, name: str) -> BaseStrategy:
        if name in self.strategies:
            strategy = self.strategies.pop(name)
            self._logger.debug(f'removed strategy {name}')
            return strategy
        else:
            raise ValueError(f"strategy {name} cannot be found, failed to remove")
    
    def _create_broker(self, bkr: tBROKER) -> BaseBroker:
        if self._env in [Environment.BACKTEST, Environment.SANDBOX]:
            from pfund.brokers.broker_simulated import SimulatedBrokerFactory
            SimulatedBroker = SimulatedBrokerFactory(bkr)
            broker = SimulatedBroker(self._env)
        else:
            BrokerClass = Broker[bkr.upper()].broker_class
            broker = BrokerClass(self._env)
        return broker

    def add_broker(self, trading_venue: tTRADING_VENUE) -> BaseBroker:
        trading_venue = TradingVenue[trading_venue.upper()]
        if trading_venue in CryptoExchange.__members__:
            bkr = Broker.CRYPTO
        # TODO: handle trading venues in DeFi
        else:
            bkr = Broker[trading_venue]
        if bkr not in self.brokers:
            broker = self._create_broker(bkr)
            self.brokers[bkr] = broker
            self._logger.debug(f'added broker {bkr}')
        return self.brokers[bkr]
    
    def remove_broker(self, bkr: tBROKER) -> BaseBroker:
        bkr = Broker[bkr.upper()]
        broker = self.brokers.pop(bkr)
        self._logger.debug(f'removed broker {bkr}')
        return broker
