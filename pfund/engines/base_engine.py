from __future__ import annotations
from typing import TYPE_CHECKING, Literal, Any
if TYPE_CHECKING:
    from pfund.datas.resolution import Resolution
    from pfund.entities.products.product_base import BaseProduct
    from pfund.entities.accounts.account_base import BaseAccount
    from pfund.datas.data_base import BaseData
    from pfund.typing import StrategyT
    from pfund.brokers.broker_base import BaseBroker
    from pfund.components.strategies.strategy_base import BaseStrategy
    from pfund.engines.engine_context import DataRangeDict

import logging
import datetime

from pfund_kit.style import cprint, RichColor, TextStyle
from pfund.components.actor_proxy import ActorProxy
from pfund.engines.settings.trade_engine_settings import TradeEngineSettings
from pfund.engines.settings.backtest_engine_settings import BacktestEngineSettings
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
    Environment.BACKTEST: TextStyle.BOLD + RichColor.BLUE,
    Environment.SANDBOX: TextStyle.BOLD + RichColor.BLACK,
    Environment.PAPER: TextStyle.BOLD + RichColor.RED,
    Environment.LIVE: TextStyle.BOLD + RichColor.GREEN,
}


class BaseEngine:
    def __init__(
        self, 
        *,
        env: Environment, 
        data_range: str | Resolution | DataRangeDict | Literal['ytd'],
        settings: TradeEngineSettings | BacktestEngineSettings | None=None,
    ):
        '''
        Args:
            data_range: range of data to be used for the engine,
                when it is a string, it is a resolution, e.g. '1m', '1d', '1w', '1mo', '1y'
                when it is a dict, it is a dict with keys 'start_date' and 'end_date', 
                    e.g. {'start_date': '2024-01-01', 'end_date': '2024-12-31'}
        '''
        from pfund.config import setup_logging
        from pfund.engines.engine_context import EngineContext
        
        env = Environment[env.upper()]
        
        # FIXME: do NOT allow LIVE env for now
        if env == Environment.LIVE:
            raise ValueError(f"{env=} is not allowed for now")
        
        setup_logging(env=env)
        self._logger: logging.Logger = logging.getLogger('pfund')
        self._context: EngineContext = EngineContext(env=env, data_range=data_range)
        self._is_running: bool = False
        self._is_gathered: bool = False
        self.brokers: dict[Broker, BaseBroker] = {}
        self.strategies: dict[str, BaseStrategy | ActorProxy[BaseStrategy]] = {}
        if settings:
            self.configure_settings(settings, persist=False)
        cprint(f"{self.name} is running (data_range=({self.data_start}, {self.data_end}))", style=ENV_COLORS[self.env])
    
    @property
    def env(self) -> Environment:
        return self._context.env
    
    @property
    def id(self) -> str:
        return self._context.id
    
    @property
    def name(self) -> str:
        return self._context.name
    
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
    
    def is_remote(self) -> bool:
        return self._context.run_mode == RunMode.REMOTE
    
    def configure_settings(self, settings: TradeEngineSettings | BacktestEngineSettings, persist: bool=False):
        '''Overrides the loaded settings with the given settings object and saves it to settings.toml

        Args:
            settings: settings object to override the current settings (if any)
        '''
        from pfund_kit.utils import toml
        from pfund import get_config

        if not isinstance(settings, (TradeEngineSettings, BacktestEngineSettings)):
            raise ValueError(f"Invalid settings type: {type(settings)}")
        
        # write settings to settings.toml
        if persist:
            config = get_config()
            env_section = self.env
            data = {env_section: settings.model_dump()}
            toml.dump(data, config.settings_file_path, mode='update', auto_inline=True)

        # update settings in context
        self._context.settings = settings
    
    def add_strategy(
        self, 
        strategy: StrategyT,
        resolution: str, 
        name: str='', 
        ray_actor_options: dict[str, Any] | None=None,
        **ray_kwargs: Any,
    ) -> StrategyT | ActorProxy[StrategyT]:
        '''
        Args:
            ray_actor_options:
                Options for Ray actor.
                will be passed to ray actor like this: Actor.options(**ray_options).remote(**ray_kwargs)
        '''
        from pfund.components.strategies.strategy_base import BaseStrategy
        
        Strategy = strategy.__class__
        StrategyName = Strategy.__name__
        assert isinstance(strategy, BaseStrategy), \
            f"strategy '{StrategyName}' is not an instance of BaseStrategy. Please create your strategy using 'class {StrategyName}(pf.Strategy)'"
        
        strat = name or strategy.name
        if strat in self.strategies:
            raise ValueError(f"{strat} already exists")

        if ray_kwargs:
            if not self.is_remote():
                from pfeed.utils.ray import setup_ray
                setup_ray()
            strategy: ActorProxy[StrategyT] = ActorProxy(strategy, name=name, ray_actor_options=ray_actor_options, **ray_kwargs)
            strategy._set_proxy(strategy)

        strategy._hydrate(
            name=strat,
            run_mode=RunMode.REMOTE if ray_kwargs else RunMode.LOCAL,
            resolution=resolution,
            engine_context=self._context
        )
        strategy._set_top_strategy()

        self.strategies[strat] = strategy  # pyright: ignore[reportArgumentType]
        self._logger.debug(f"added '{strat}'")
        return strategy
    
    def get_strategy(self, name: str) -> BaseStrategy | ActorProxy[BaseStrategy]:
        return self.strategies[name]
    
    def _add_broker(self, trading_venue: TradingVenue) -> BaseBroker:
        from pfund.brokers import create_broker
        bkr: Broker = TradingVenue[trading_venue.upper()].broker
        if bkr not in self.brokers:
            broker = create_broker(env=self.env, bkr=bkr, settings=self.settings)
            self.brokers[bkr] = broker
            self._logger.debug(f'added broker {bkr}')
        return self.brokers[bkr]
    
    def get_broker(self, bkr: Broker) -> BaseBroker:
        return self.brokers[Broker[bkr.upper()]]
    
    def _add_product(self, product: BaseProduct):
        broker: BaseBroker = self._add_broker(product.trading_venue)
        broker.add_product(exch=product.exch, basis=str(product.basis), name=product.name, symbol=product.symbol, **product.specs)
        self._logger.debug(f'added product {product.symbol}')
    
    def _add_account(self, account: BaseAccount):
        broker: BaseBroker = self._add_broker(account.trading_venue)
        account = broker.add_account(**account.to_dict())
        self._logger.debug(f'added account {account}')
    
    def _gather(self):
        '''
        Sets up everything before run.
        - updates zmq ports in settings
        '''
        from pfund.datas.data_market import MarketData
        if not self._is_gathered:
            for strategy in self.strategies.values():
                strategy: BaseStrategy | ActorProxy[BaseStrategy]
                strategy._gather()
                
                accounts: list[BaseAccount] = strategy.get_accounts()
                for account in accounts:
                    self._add_account(account)
                
                datas: list[BaseData] = strategy._get_datas_in_use()
                for data in datas:
                    if isinstance(data, MarketData):
                        self._add_product(data.product)
                    else:
                        if hasattr(data, 'product'):
                            raise NotImplementedError(f"Unhandled data type that has product attribute: {type(data)}. It should also call add_product().")
        else:
            self._logger.debug(f'{self.name} is already gathered')
    
    def run(self):
        if not self.is_running():
            self._is_running = True
            self._gather()
            self._is_gathered = True
            for broker in self.brokers.values():
                broker.start()
            for strategy in self.strategies.values():
                strategy.start()
        else:
            self._logger.debug(f'{self.name} is already running')

    def end(self):
        from pfeed.utils.ray import shutdown_ray
        if self.is_running():
            self._is_running = False
            for strategy in self.strategies.values():
                strategy.stop()
            for broker in self.brokers.values():
                broker.stop()
        else:
            self._logger.debug(f'{self.name} is not running')
        shutdown_ray()