from __future__ import annotations
from typing import TYPE_CHECKING, Literal, overload, ClassVar
if TYPE_CHECKING:
    from mtflow.stores.mtstore import MTStore
    from mtflow.kernel import TradeKernel
    from pfeed.typing import tDATA_TOOL
    from pfund.typing import StrategyT, tENVIRONMENT, tBROKER, DataRangeDict, tDATABASE
    from pfund.typing import ExternalListenersDict
    from pfund.brokers.broker_trade import BaseBroker
    from pfund.brokers.broker_crypto import CryptoBroker
    from pfund.brokers.ib.broker_ib import IBBroker
    from pfund.strategies.strategy_base import BaseStrategy
    from pfund.models.model_base import BaseModel
    from pfund.engines.trade_engine_settings import TradeEngineSettings
    from pfund.engines.backtest_engine_settings import BacktestEngineSettings

import logging
import datetime
import importlib

from pfeed.enums import DataTool
from pfund import cprint
from pfund.engines.meta_engine import MetaEngine
from pfund.enums import Environment, Broker, RunMode, Database
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
    _initialized: ClassVar[bool] = False
    _env: ClassVar[Environment]
    _run_mode: ClassVar[RunMode]
    data_tool: ClassVar[DataTool]
    dataset_start: ClassVar[datetime.date]
    dataset_end: ClassVar[datetime.date]
    database: ClassVar[tDATABASE | None]
    settings: ClassVar[TradeEngineSettings | BacktestEngineSettings]
    external_listeners: ClassVar[ExternalListenersDict | None]
    
    _logger: logging.Logger
    _store: MTStore
    _kernel: TradeKernel | None
    _brokers: dict[str, BaseBroker]
    _strategies: dict[str, BaseStrategy]
    
        
    def __init__(
        self, 
        *,
        env: tENVIRONMENT, 
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
        if not hasattr(cls, "_initialized"):
            env = Environment[env.upper()]
            cls._env = env
            if is_wasm():
                cls._run_mode = RunMode.WASM
                assert not use_ray, 'Ray is not supported in WASM mode'
            else:
                cls._run_mode = RunMode.REMOTE if use_ray else RunMode.LOCAL
            cls.data_tool = DataTool[data_tool.lower()]
            cls.database = Database[database.upper()] if database else None
            cls.settings = settings
            cls.external_listeners = ExternalListeners(**(external_listeners or {}))
            if cls.external_listeners.recorder is False and cls.database is not None:
                cprint(
                    'WARNING: `database` is set but recorder is disabled in `external_listeners`, '
                    'data will be written to the database by the engine itself, which will introduce latency',
                    style='bold yellow'
                )
            is_data_range_dict = isinstance(data_range, dict)
            cls.dataset_start, cls.dataset_end = TimeBasedFeed._parse_date_range(
                start_date=data_range['start_date'] if is_data_range_dict else '',
                end_date=data_range.get('end_date', '') if is_data_range_dict else '',
                rollback_period=data_range if not is_data_range_dict else '',
            )
            cls._initialized = True
            type(cls).lock()  # Locks any future class modifications


        # FIXME:
        if env == Environment.BACKTEST:
            assert cls.__name__ == 'BacktestEngine', f'{env=} is only allowed to be created using BacktestEngine'
        cprint(f"{env} Engine is running", style=ENV_COLORS[env])
        
        
        # FIXME: do NOT allow LIVE env for now
        assert env != Environment.LIVE, f"{env=} is not allowed for now"
        
        
        self._setup_logging()
        self._logger = logging.getLogger('pfund')
        self._store = MTStore(env=cls._env, data_tool=cls.data_tool)
        self._kernel = TradeKernel(
            mode=cls._run_mode,
            database=cls.database,
            external_listeners=cls.external_listeners,
            zmq_urls=cls.settings.zmq_urls,
            zmq_ports=cls.settings.zmq_ports,
        )
        self._brokers = {}
        self._strategies = {}
        
    @property
    def brokers(self) -> dict[str, BaseBroker]:
        return self._brokers
    
    @property
    def strategies(self) -> dict[str, BaseStrategy]:
        return self._strategies
    
    def _setup_logging(self):
        from pfund._logging import setup_loggers
        from pfund import get_config
        config = get_config()
        log_path = f'{config.log_path}/{self._env.lower()}'
        logging_config_file_path = config.logging_config_file_path
        logging_configurator  = setup_loggers(log_path, logging_config_file_path, user_logging_config=config.logging_config)
        config.set_logging_configurator(logging_configurator)
    
    def _setup_ray_actors(self, auto_wrap: bool=True):
        def _check_if_add_ray_actor(component: BaseStrategy | BaseModel):
            if component.is_ray_actor():
                component._setup_zmq()
                self._kernel.add_ray_actor(component, auto_wrap=auto_wrap)
            for subcomponent in component.components:
                _check_if_add_ray_actor(subcomponent)
        for strategy in self._strategies.values():
            _check_if_add_ray_actor(strategy)

    def run(self):
        self._store.freeze()
        if self._kernel is not None and self._run_mode == RunMode.REMOTE:
            self._setup_ray_actors()
    
    def is_running(self) -> bool:
        return self._kernel._is_running
    
    def get_strategy(self, strat: str) -> BaseStrategy | None:
        return self._strategies.get(strat, None)

    def add_strategy(self, strategy: StrategyT, resolution: str, name: str='') -> StrategyT:        
        assert isinstance(strategy, BaseStrategy), \
            f"strategy '{strategy.__class__.__name__}' is not an instance of BaseStrategy. Please create your strategy using 'class {strategy.__class__.__name__}(BaseStrategy)'"
        if name:
            strategy._set_name(name)
        strategy._create_logger()
        strategy._set_resolution(resolution)
        strat = strategy.name
        if strat in self._strategies:
            raise ValueError(f"{strategy.name} already exists")
        else:
            self._strategies[strat] = strategy
            self._logger.debug(f"added '{strategy.name}'")
            return strategy
    
    def remove_strategy(self, strat: str):
        if strat in self._strategies:
            del self._strategies[strat]
            self._logger.debug(f'removed strategy {strat}')
        else:
            self._logger.error(f'strategy {strat} cannot be found, failed to remove')

    # conditional typing, returns the exact type of broker
    @overload
    def get_broker(self, bkr: Literal['CRYPTO']) -> CryptoBroker: ...
        
    # conditional typing, returns the exact type of broker
    @overload
    def get_broker(self, bkr: Literal['IB']) -> IBBroker: ...

    def get_broker(self, bkr: tBROKER) -> BaseBroker:
        return self._brokers[bkr.upper()]
    
    def add_broker(self, bkr: str) -> BaseBroker:
        bkr = bkr.upper()
        if bkr in self.brokers:
            return self.get_broker(bkr)
        broker = self._create_broker(bkr)
        self._brokers[bkr] = broker
        self._logger.debug(f'added broker {bkr}')
        return broker
    
    def remove_broker(self, bkr: tBROKER) -> BaseBroker:
        broker = self._brokers.pop(bkr.upper())
        self._logger.debug(f'removed broker {bkr}')
        return broker
    
    def get_Broker(self, bkr: tBROKER) -> type[BaseBroker]:
        broker = Broker[bkr.upper()]
        if broker == Broker.CRYPTO:
            BrokerClass = getattr(importlib.import_module('pfund.brokers.broker_crypto'), 'CryptoBroker')
        elif broker == Broker.IB:
            BrokerClass = getattr(importlib.import_module('pfund.brokers.ib.broker_ib'), 'IBBroker')
        else:
            raise ValueError(f'broker {bkr} is not supported')
        return BrokerClass