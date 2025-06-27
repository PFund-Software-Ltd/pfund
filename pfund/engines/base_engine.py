from __future__ import annotations
from typing import TYPE_CHECKING, Literal, ClassVar
if TYPE_CHECKING:
    from mtflow.stores.mtstore import MTStore
    from mtflow.kernel import TradeKernel
    from pfund.accounts.account_crypto import CryptoAccount
    from pfund.accounts.account_ib import IBAccount
    from mtflow.stores.trading_store import TradingStore
    from pfund.messenger import Messenger
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
        ComponentName,
    )
    from pfund.brokers.broker_base import BaseBroker
    from pfund.strategies.strategy_base import BaseStrategy
    from pfund.engines.base_engine_settings import BaseEngineSettings

import time
import logging
import datetime

from pfeed.enums import DataTool
from pfund import cprint, get_config
from pfund.engines.meta_engine import MetaEngine
from pfund.proxies.actor_proxy import ActorProxy
from pfund._logging.buffer import LogBuffer
from pfund.enums import (
    Environment, 
    Broker, 
    RunMode, 
    Database, 
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
config = get_config()


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
    _logging_config: ClassVar[dict]
    
    name: str
    _logger: logging.Logger
    _kernel: TradeKernel
    _is_running: bool
    _messenger: Messenger | None
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
        from pfund.messenger import Messenger
        
        cls = self.__class__
        if not hasattr(cls, "_initialized"):
            from pfeed.feeds.time_based_feed import TimeBasedFeed
            from pfund.external_listeners import ExternalListeners
            from pfund.utils.utils import derive_run_mode

            env = Environment[env.upper()]
            config._load_env_file(env)
            
            cls._env = env
            cls._logging_config = cls._setup_logging()
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
        assert cls._env != Environment.LIVE, f"{cls._env=} is not allowed for now"

        
        self.name = name or self._get_default_name()
        if 'engine' not in self.name.lower():
            self.name += '_engine'
        self._logger: logging.Logger = logging.getLogger('pfund')
        self._kernel = TradeKernel(
            database=cls._database,
            data_tool=cls._data_tool,
            use_deltalake=config.use_deltalake,
            external_listeners=cls._external_listeners,
        )
        if not self.is_wasm():
            self._messenger = Messenger(
                zmq_url=cls._settings.zmq_urls.get(self.name, ''), 
                zmq_ports=cls._settings.zmq_ports,
            )
        else:
            self._messenger = None
        self.brokers: dict[Broker, BaseBroker] = {}
        self.strategies: dict[str, BaseStrategy | ActorProxy] = {}
        self._is_running: bool = False
        self._is_gathered: bool = False
    
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
    
    def is_wasm(self):
        return self._run_mode == RunMode.WASM
    
    @classmethod
    def _setup_logging(cls) -> dict:
        from pfund._logging import setup_logging_config
        from pfund._logging.config import LoggingDictConfigurator
        config = get_config()
        log_path = f'{config.log_path}/{cls._env}'
        user_logging_config = config.logging_config
        logging_config_file_path = config.logging_config_file_path
        logging_config = setup_logging_config(log_path, logging_config_file_path, user_logging_config=user_logging_config)
        # ≈ logging.config.dictConfig(logging_config) with a custom configurator
        logging_configurator = LoggingDictConfigurator(logging_config)
        logging_configurator.configure()
        return logging_config
    
    def gather(self):
        '''
        Sets up everything before run.
        - updates zmq ports in settings
        - registers components, data to mtstore
        - freezes mtstore.
        This method can be called by user before run to do some custom setup,
        e.g. get pfeed's dataflows and add custom transformations to them by calling:
        # TODO:
        engine.gather()
        dataflow = engine.get_dataflow(...)
        dataflow.add_transformation(...)
        engine.run()
        '''
        if not self._is_gathered:
            cls = self.__class__
            # TODO: pfeed's dataflows should be created in data engine at this level
            # TODO: ws_groups in engine settings? used to set ws product groups in pfeed
            for strategy in self.strategies.values():
                strategy: BaseStrategy | ActorProxy
                strategy._gather()
                cls._settings.zmq_ports.update(strategy._get_zmq_ports_in_use())
                metadata = strategy.to_dict(stringify=False)
                self._register_component(metadata)
            self._is_gathered = True
        else:
            self._logger.debug(f'{self.name} is already gathered')
    
    def run(self):
        if not self.is_running():
            self._is_running = True
            self.gather()
            self._kernel.run()
            for strategy in self.strategies.values():
                strategy.start()

            # use ZeroMQ as long as not in WASM mode
            if not self.is_wasm():
                self._messenger.subscribe()
                self._messenger.start()
                while self.is_running():
                    # TODO: should update positions, balances, orders etc. using proxy
                    if msg := self._messenger._proxy.recv():
                        channel, topic, data, pub_ts = msg
                        self._logger.debug(f'{channel} {topic} {data} {pub_ts}')
                    if msg := self._messenger._router.recv():
                        channel, topic, data, pub_ts = msg
                        self._logger.debug(f'{channel} {topic} {data} {pub_ts}')
                    if msg := self._messenger._publisher.recv():
                        channel, topic, data, pub_ts = msg
                        self._logger.debug(f'{channel} {topic} {data} {pub_ts}')
                    time.sleep(0.0001)
            else:
                # TODO: get msg from data engine
                msg = ...
                for strategy in self.strategies.values():
                    strategy.databoy._collect(msg)
        else:
            raise RuntimeError('Engine is already running')
    
    # TODO:
    def end(self):
        self._is_running = False
        self._kernel.end()
        
    def is_running(self) -> bool:
        return self._is_running
    
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
        strategy._hydrate(
            env=self._env,
            name=strat,
            run_mode=run_mode,
            resolution=resolution,
            engine=None if is_remote else self,
            settings=self._settings,
            logging_config=self._logging_config,
            data_tool=self._data_tool,
            storage=config.storage,
            storage_options=config.storage_options,
            use_deltalake=config.use_deltalake,
        )
        strategy._set_top_strategy(True)

        self.strategies[strat] = strategy
        self._logger.debug(f"added '{strat}'")
        return strategy
    
    def get_strategy(self, name: str) -> BaseStrategy | ActorProxy:
        return self.strategies[name]
    
    def _add_broker(self, trading_venue: TradingVenue | tTradingVenue) -> BaseBroker:
        from pfund.brokers import create_broker
        bkr: Broker = TradingVenue[trading_venue.upper()].broker
        if bkr not in self.brokers:
            broker = create_broker(env=self._env, bkr=bkr)
            self.brokers[bkr] = broker
            self._logger.debug(f'added broker {bkr}')
        return self.brokers[bkr]
    
    def get_broker(self, bkr: tBroker) -> BaseBroker:
        return self.brokers[bkr.upper()]
    
    def _register_component(self, component_metadata: dict):
        component_name = component_metadata['name']
        consumer_names = component_metadata['consumers']
        self._kernel.register_component(consumer_names, component_name, component_metadata)
        datas: list[TimeBasedData] = component_metadata['datas']
        for data in datas:
            if not data.is_resamplee():
                broker: BaseBroker = self.get_broker(data.product.bkr)
                broker._add_data_channel(data)
            self._register_product(data.product)
        for account in component_metadata.get('accounts', []):
            self._register_account(account)
        # register sub-components (nested components)
        strategies: list[dict] = component_metadata.get('strategies', [])
        models: list[dict] = component_metadata.get('models', [])
        features: list[dict] = component_metadata.get('features', [])
        indicators: list[dict] = component_metadata.get('indicators', [])
        components: list[dict] = strategies + models + features + indicators
        for component_metadata in components:
            self._register_component(component_metadata)
        
    def _register_product(self, product: BaseProduct):
        broker: BaseBroker = self._add_broker(product.trading_venue)
        if broker.name == Broker.CRYPTO:
            broker.add_product(exch=product.exch, basis=str(product.basis), name=product.name, symbol=product.symbol, **product.specs)
        elif broker.name == Broker.IB:
            broker.add_product(exch=product.exch, basis=str(product.basis), name=product.name, symbol=product.symbol, **product.specs)
        else:
            raise NotImplementedError(f"Broker {broker.name} is not supported")
        self._logger.debug(f'added product {product.symbol}')
    
    def _register_account(self, account: BaseAccount) -> BaseAccount:
        broker: BaseBroker = self._add_broker(account.trading_venue)
        if broker.name == Broker.CRYPTO:
            account: CryptoAccount
            account = broker.add_account(exch=account.trading_venue, name=account.name, key=account._key, secret=account._secret)
        elif broker.name == Broker.IB:
            account: IBAccount
            account = broker.add_account(name=account.name, host=account._host, port=account._port, client_id=account._client_id)
        else:
            raise NotImplementedError(f"Broker {broker.name} is not supported")
        self._logger.debug(f'added account {account}')
        return account
    