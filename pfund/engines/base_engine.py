from __future__ import annotations
from typing import TYPE_CHECKING, Literal
if TYPE_CHECKING:
    from pfund.datas.resolution import Resolution
    from pfund.products.product_base import BaseProduct
    from pfund.accounts.account_base import BaseAccount
    from pfund.datas.data_time_based import TimeBasedData
    from pfund.typing import (
        StrategyT,
        tBroker,
        DataParamsDict,
    )
    from pfund.brokers.broker_base import BaseBroker
    from pfund.strategies.strategy_base import BaseStrategy
    from pfund.settings import TradeEngineSettings, BacktestEngineSettings

import logging
import datetime

from pfund import get_config
from pfund.proxies.actor_proxy import ActorProxy
from pfund.proxies.engine_proxy import EngineProxy
from pfund.engines.engine_context import EngineContext, DataRangeDict
from pfund.enums import (
    Environment,
    Broker,
    RunMode,
    TradingVenue,
)


ENV_COLORS = {
    # 'yellow': 'bold yellow on #ffffe0',
    # 'magenta': 'bold magenta on #fff0ff',
    # 'TRAIN': 'bold cyan on #d0ffff',
    Environment.BACKTEST: 'bold blue',
    Environment.SANDBOX: 'bold black',
    Environment.PAPER: 'bold red',
    Environment.LIVE: 'bold green',
}
config = get_config()



class BaseEngine:
    def __init__(
        self, 
        *,
        env: Environment, 
        data_range: str | Resolution | DataRangeDict | Literal['ytd'],
        name: str='',
    ):
        '''
        Args:
            data_range: range of data to be used for the engine,
                when it is a string, it is a resolution, e.g. '1m', '1d', '1w', '1mo', '1y'
                when it is a dict, it is a dict with keys 'start_date' and 'end_date', 
                    e.g. {'start_date': '2024-01-01', 'end_date': '2024-12-31'}
        '''
        from pfund.config import setup_logging
        from pfund_kit.style import cprint
        
        env = Environment[env.upper()]
        
        # FIXME: do NOT allow LIVE env for now
        if env == Environment.LIVE:
            raise ValueError(f"{env=} is not allowed for now")
        
        setup_logging(env=env)
        self._logger = logging.getLogger('pfund')
        self.name = self._get_default_name()
        if name:
            self._set_name(name)
        self._is_running: bool = False
        self._is_gathered: bool = False
        self.brokers: dict[Broker, BaseBroker] = {}
        self.strategies: dict[str, BaseStrategy | ActorProxy] = {}
        # TODO: add risk engine?
        # self._risk_engine = RiskEngine()  
        self._context: EngineContext = EngineContext(env=env, data_range=data_range)
        cprint(f"{self.env} {self.name} is running (data_range=({self.data_start}, {self.data_end}))", style=ENV_COLORS[self.env])
    
    @property
    def env(self) -> Environment:
        return self._context.env

    @property
    def settings(self) -> TradeEngineSettings | BacktestEngineSettings:
        return self._context.settings
    
    @property
    def data_start(self) -> datetime.date:
        return self._context.data_start
    
    @property
    def data_end(self) -> datetime.date:
        return self._context.data_end
    
    def is_running(self) -> bool:
        return self._is_running
    
    def _get_default_name(self) -> str:
        return f"{self.__class__.__name__}"
    
    def _set_name(self, name: str):
        if not name:
            return
        self.name = name
        if not self.name.lower().endswith("engine"):
            self.name += "_engine"
    
    def configure_settings(self, settings: TradeEngineSettings | BacktestEngineSettings):
        '''Overrides the loaded settings with the given settings object and saves it to settings.toml

        Args:
            settings: settings object to override the current settings (if any)
        '''
        from pfund_kit.utils import toml

        # write settings to settings.toml
        env_section = self.env
        data = {env_section: settings.model_dump()}
        toml.dump(data, config.settings_file_path, mode='update', auto_inline=True)

        # update settings in config
        self._context.settings = settings
    
    # FIXME
    def get_data_params(self) -> DataParamsDict:
        '''Data params are used in components' data stores'''
        return {
            'data_start': self._data_start,
            'data_end': self._data_end,
            'data_tool': self._data_tool,
            # FIXME
            'storage': config.storage,
            'storage_options': config.storage_options,
            'use_deltalake': config.use_deltalake,
        }
        
    # TODO: create EngineMetadata class (typed dict/dataclass/pydantic model)
    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'env': self.env.value,
            'data_start': self.data_start.strftime('%Y-%m-%d'),
            'data_end': self.data_end.strftime('%Y-%m-%d'),
            'settings': self.settings.model_dump(),
        }
    
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
        from pfund.utils import derive_run_mode
        
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
            name=strat,
            run_mode=run_mode,
            resolution=resolution,
            engine=EngineProxy.from_engine(self) if is_remote else self,
        )
        strategy._set_top_strategy(True)

        self.strategies[strat] = strategy
        self._logger.debug(f"added '{strat}'")
        return strategy
    
    def get_strategy(self, name: str) -> BaseStrategy | ActorProxy:
        return self.strategies[name]
    
    def _add_broker(self, trading_venue: TradingVenue) -> BaseBroker:
        from pfund.brokers import create_broker
        bkr: Broker = TradingVenue[trading_venue.upper()].broker
        if bkr not in self.brokers:
            broker = create_broker(env=self.env, bkr=bkr)
            self.brokers[bkr] = broker
            self._logger.debug(f'added broker {bkr}')
        return self.brokers[bkr]
    
    def get_broker(self, bkr: tBroker) -> BaseBroker:
        return self.brokers[bkr.upper()]
    
    def _register_component(self, component_metadata: dict):
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
        elif broker.name == Broker.IBKR:
            broker.add_product(exch=product.exch, basis=str(product.basis), name=product.name, symbol=product.symbol, **product.specs)
        else:
            raise NotImplementedError(f"Broker {broker.name} is not supported")
        self._logger.debug(f'added product {product.symbol}')
    
    def _register_account(self, account: BaseAccount):
        broker: BaseBroker = self._add_broker(account.trading_venue)
        if broker.name == Broker.CRYPTO:
            account = broker.add_account(exch=account.trading_venue, name=account.name, key=account._key, secret=account._secret)
        elif broker.name == Broker.IBKR:
            account = broker.add_account(name=account.name, host=account._host, port=account._port, client_id=account._client_id)
        else:
            raise NotImplementedError(f"Broker {broker.name} is not supported")
        self._logger.debug(f'added account {account}')
    
    def gather(self):
        '''
        Sets up everything before run.
        - updates zmq ports in settings
        - registers components, data to mtstore
        - freezes mtstore.
        '''
        if not self._is_gathered:
            # TODO: add engine metadata to mtflow
            engine_metadata = self.to_dict()

            for strategy in self.strategies.values():
                strategy: BaseStrategy | ActorProxy
                strategy._gather()
                
                # updates zmq ports in settings
                self._context.settings.zmq_ports.update(strategy._get_zmq_ports_in_use())
                
                # registers accounts
                accounts: list[BaseAccount] = strategy.get_accounts()
                for account in accounts:
                    self._register_account(account)
                
                # registers products
                datas: list[TimeBasedData] = strategy._get_datas_in_use()
                for data in datas:
                    self._register_product(data.product)
                
                # registers components
                metadata = strategy.to_dict()
                self._register_component(metadata)
        else:
            self._logger.debug(f'{self.name} is already gathered')
    
    def run(self):
        if not self.is_running():
            self._is_running = True
            self.gather()
            # TODO: start brokers
            # for broker in self.brokers.values():
            #     broker.start()
            for strategy in self.strategies.values():
                strategy.start()
        else:
            self._logger.debug(f'{self.name} is already running')

    def end(self):
        if self.is_running():
            self._is_running = False
            for strategy in self.strategies.values():
                strategy.stop()
            for broker in self.brokers.values():
                broker.stop()
        else:
            self._logger.debug(f'{self.name} is not running')
