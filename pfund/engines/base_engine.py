# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
from __future__ import annotations
from typing import TYPE_CHECKING, Literal, Any
if TYPE_CHECKING:
    from pfeed.sources.pfund.engine_feed import EngineFeed
    from pfund.datas.resolution import Resolution
    from pfund.entities.products.product_base import BaseProduct
    from pfund.entities.accounts.account_base import BaseAccount
    from pfund.datas.data_base import BaseData
    from pfund.typing import StrategyT
    from pfund.brokers.broker_base import BaseBroker
    from pfund.components.strategies.strategy_base import BaseStrategy
    from pfund.engines.engine_context import DataRangeDict
    from pfund.components.actor_proxy import ActorProxy
    from pfund.engines.settings.trade_engine_settings import TradeEngineSettings
    from pfund.engines.settings.sandbox_engine_settings import SandboxEngineSettings
    from pfund.engines.settings.backtest_engine_settings import BacktestEngineSettings

import logging
import datetime

from pfeed.enums import DataCategory
from pfund_kit.style import cprint, RichColor, TextStyle
from pfund.enums import (
    Environment,
    Broker,
    RunMode,
    TradingVenue,
    ComponentType,
)


ENV_COLORS = {
    # 'yellow': 'bold yellow on #ffffe0',
    # 'magenta': 'bold magenta on #fff0ff',
    # 'TRAIN': 'bold cyan on #d0ffff',
    Environment.BACKTEST: TextStyle.BOLD + RichColor.BLUE,
    Environment.SANDBOX: TextStyle.BOLD + RichColor.GREY0,
    Environment.PAPER: TextStyle.BOLD + RichColor.RED,
    Environment.LIVE: TextStyle.BOLD + RichColor.GREEN,
}


class BaseEngine:
    def __init__(
        self, 
        *,
        env: Environment, 
        name: str,
        data_range: str | Resolution | DataRangeDict | Literal['ytd'],
        settings: TradeEngineSettings | BacktestEngineSettings | SandboxEngineSettings | None=None,
    ):
        '''
        Args:
            data_range: range of data to be used for the engine,
                when it is a string, it is a resolution, e.g. '1m', '1d', '1w', '1mo', '1y'
                when it is a dict, it is a dict with keys 'start_date' and 'end_date', 
                    e.g. {'start_date': '2024-01-01', 'end_date': '2024-12-31'}
        '''
        import pfeed as pe
        from pfund.config import setup_logging
        from pfund.engines.engine_context import EngineContext
        
        env = Environment[env.upper()]
        
        # FIXME: do NOT allow LIVE env for now
        if env == Environment.LIVE:
            raise ValueError(f"{env=} is not allowed for now")
        
        setup_logging(env=env, engine_name=name)
        self._logger: logging.Logger = logging.getLogger('pfund')
        self._context: EngineContext = EngineContext(env=env, name=name, data_range=data_range, settings=settings)
        self._is_running = False
        # TODO: write engine's states using engine_feed.load()
        self._feed: EngineFeed = pe.PFund().engine_feed
        self.brokers: dict[Broker, BaseBroker] = {}
        self.strategies: dict[str, BaseStrategy | ActorProxy[BaseStrategy]] = {}
        self.results: dict[str, Any] | None = None
        cprint(f"{self.env} {self.name} is running (data_range=({self.data_start}, {self.data_end}))", style=ENV_COLORS[self.env])
    
    @property
    def env(self) -> Environment:
        return self._context.env
    
    @property
    def name(self) -> str:
        return self._context.name
    
    @property
    def run_mode(self) -> RunMode:
        return self._context.run_mode
    
    @property
    def settings(self) -> TradeEngineSettings | BacktestEngineSettings | SandboxEngineSettings:
        return self._context.settings
    
    @property
    def data_start(self) -> datetime.date:
        return self._context.data_start
    
    @property
    def data_end(self) -> datetime.date:
        return self._context.data_end
    
    def is_running(self) -> bool:
        return self._is_running
    
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
        from pfund.components.actor_proxy import ActorProxy
        from pfund.components.strategies.strategy_base import BaseStrategy
        
        Strategy = strategy.__class__
        StrategyName = Strategy.__name__
        assert isinstance(strategy, BaseStrategy), \
            f"strategy '{StrategyName}' is not an instance of BaseStrategy. Please create your strategy using 'class {StrategyName}(pf.Strategy)'"
        
        strat = name or strategy.name
        if strat in self.strategies:
            raise ValueError(f"{strat} already exists")

        if ray_kwargs:
            strategy: ActorProxy[StrategyT] = ActorProxy(
                strategy, 
                name=strat,
                resolution=resolution,
                component_type=ComponentType.strategy,
                engine_context=self._context,
                ray_actor_options=ray_actor_options, 
                **ray_kwargs
            )

        strategy._hydrate(
            name=strat,
            run_mode=RunMode.REMOTE if ray_kwargs else RunMode.LOCAL,
            resolution=resolution,
            engine_context=self._context,
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

    def _get_all_datas(self) -> set[BaseData]:
        datas: list[BaseData] = []
        for strategy in self.strategies.values():
            datas.extend(strategy.get_datas())
            for component in strategy.get_components():
                datas.extend(component.get_datas())
        return set(datas)
    
    def _gather(self):
        for strategy in self.strategies.values():
            strategy: BaseStrategy | ActorProxy[BaseStrategy]
            strategy._gather()
            
            accounts: list[BaseAccount] = strategy.get_accounts()
            for account in accounts:
                self._add_account(account)
            
        for data in self._get_all_datas():
            if data.category == DataCategory.MARKET_DATA:
                self._add_product(data.product)
            else:
                raise NotImplementedError(f"Unhandled data type: {type(data)}")
    
    def run(self):
        self._logger.debug(f'Running {self.name}...')
        self._is_running = True
        self._gather()
        for broker in self.brokers.values():
            broker.start()
        for strategy in self.strategies.values():
            strategy.start()

    def end(self):
        from pfeed.utils.ray import shutdown_ray
        self._logger.debug(f'Ending {self.name}...')
        for strategy in self.strategies.values():
            strategy.stop()
        for broker in self.brokers.values():
            broker.stop()
        self._is_running = False
        shutdown_ray()