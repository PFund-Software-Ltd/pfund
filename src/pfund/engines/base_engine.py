# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportArgumentType=false
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    from mtflow.contexts.base_context import BaseContext
    from pfeed.sources.pfund.engine_feed import PFundEngineFeed

    from pfund.components.actor_proxy import ActorProxy
    from pfund.venues.venue_base import AnyVenue
    from pfund.venues.venue_config import VenueConfig
    from pfund.components.strategies.strategy_base import BaseStrategy
    from pfund.datas.data_base import BaseData
    from pfund.datas.data_market import MarketData
    from pfund.datas.resolution import Resolution
    from pfund.engines.engine_context import DataRangeDict
    from pfund.engines.settings.base_engine_settings import BaseEngineSettings
    from pfund.entities import BaseAccount, BaseProduct
    from pfund.typing import StrategyT, ComponentName

import logging

from pfund_kit.style import cprint
from pfund_kit.utils.singleton import SingletonMeta
from pfeed.enums import DataCategory

from pfund.managers import OrderManager, PortfolioManager, RiskManager
from pfund.enums import (
    ComponentType,
    Environment,
    TradingVenue,
    RunMode,
)


class BaseEngine(metaclass=SingletonMeta):
    def __init__(
        self,
        *,
        env: Environment,
        name: str,
        data_range: str | Resolution | DataRangeDict | tuple[str, str] | Literal["ytd"],
        settings: BaseEngineSettings | None = None,
    ):
        """
        Args:
            name: engine name
            data_range: range of data to be used for the engine,
                when it is a string, it is a resolution, e.g. '1m', '1d', '1w', '1mo', '1y'
                when it is a dict, it is a dict with keys 'start_date' and 'end_date',
                    e.g. {'start_date': '2024-01-01', 'end_date': '2024-12-31'}
                when it is a tuple, it is (start_date, end_date),
                    e.g. ('2024-01-01', '2024-12-31')
        """
        import pfeed as pe

        from pfund.config import setup_logging
        from pfund.engines.engine_context import EngineContext

        env = Environment[env.upper()]

        # FIXME: do NOT allow LIVE env for now
        if env == Environment.LIVE:
            raise ValueError(f"{env=} is not allowed for now")

        setup_logging(env=env, engine_name=name)
        self._logger: logging.Logger = logging.getLogger("pfund")
        self._context: EngineContext = EngineContext(
            env=env,
            name=name,
            data_range=data_range,
            settings=settings,
        )
        self._is_running = False
        self._feed: PFundEngineFeed = pe.PFund().engine_feed
        self._strategies: dict[
            ComponentName, BaseStrategy | ActorProxy[BaseStrategy]
        ] = {}
        self._venues: dict[TradingVenue, AnyVenue] = {}
        self._order_manager = OrderManager()
        self._portfolio_manager = PortfolioManager()
        self._risk_manager = RiskManager()

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
    def settings(self) -> BaseEngineSettings:
        return self._context.settings

    @property
    def order_manager(self) -> OrderManager:
        return self._order_manager

    om = order_manager

    @property
    def portfolio_manager(self) -> PortfolioManager:
        return self._portfolio_manager

    pm = portfolio_manager

    @property
    def risk_manager(self) -> RiskManager:
        return self._risk_manager

    rm = risk_manager

    def is_running(self) -> bool:
        return self._is_running

    def _add_product(self, product: BaseProduct):
        for _venue in self._venues.values():
            if (existing := _venue.products.get(product.name)) is not None:
                raise ValueError(
                    f'product name "{product.name}" is already used by {existing!r}; '
                    + "product names must be unique across the engine"
                )
        venue: AnyVenue = self.add_venue(product.source)
        venue.add_product(product)

    def _add_account(self, account: BaseAccount):
        for _venue in self._venues.values():
            if (existing := _venue.accounts.get(account.name)) is not None:
                raise ValueError(
                    f'account name "{account.name}" is already used by {existing!r}; '
                    + "account names must be unique across the engine"
                )
        if account.env != self.env:
            raise ValueError(
                f"account env {account.env} does not match engine env {self.env}"
            )
        venue: AnyVenue = self.add_venue(account.venue)
        venue.add_account(account)

    def add_venue(
        self, venue: TradingVenue | str, config: VenueConfig | None = None
    ) -> AnyVenue:
        venue = TradingVenue[venue.upper()]
        if venue not in self._venues:
            VenueClass = venue.venue_class
            self._venues[venue] = VenueClass(env=self.env, config=config)
            self._logger.debug(f"added trading venue {venue}")
        elif config is not None:
            raise ValueError(f"{venue} already exists and cannot be configured")
        return self._venues[venue]

    def get_venue(self, venue: TradingVenue | str) -> AnyVenue:
        venue = TradingVenue[venue.upper()]
        return self._venues[venue]

    def add_strategy(
        self,
        strategy: StrategyT,
        resolution: str,
        name: str = "",
        ray_actor_options: dict[str, Any] | None = None,
        **ray_kwargs: Any,
    ) -> StrategyT | ActorProxy[StrategyT]:
        """
        Args:
            ray_actor_options:
                Options for Ray actor.
                will be passed to ray actor like this: Actor.options(**ray_options).remote(**ray_kwargs)
        """
        from pfund.components.actor_proxy import ActorProxy
        from pfund.components.strategies.strategy_base import BaseStrategy

        Strategy = strategy.__class__
        StrategyName = Strategy.__name__
        assert isinstance(strategy, BaseStrategy), (
            f"strategy '{StrategyName}' is not an instance of BaseStrategy. Please create your strategy using 'class {StrategyName}(pf.Strategy)'"
        )

        strat = name or strategy.name
        if strat in self._strategies:
            raise ValueError(f"{strat} already exists")

        # enforce GLOBAL name uniqueness (across other Ray actors too), not just this engine's dict
        if ray_kwargs:
            # upgrade BEFORE the actor is created, so the shared-registry context is what ships into it
            from pfund.engines.component_registry import to_registry_proxy

            self._context.component_registry = to_registry_proxy(
                self._context.component_registry
            )
        # claim before spawning the actor, so a duplicate name aborts without leaking a live actor
        self._context.component_registry.claim(strat)

        if ray_kwargs:
            strategy: ActorProxy[StrategyT] = ActorProxy(
                strategy,
                name=strat,
                resolution=resolution,
                component_type=ComponentType.strategy,
                engine_context=self._context,
                ray_actor_options=ray_actor_options,
                **ray_kwargs,
            )

        strategy._hydrate(
            name=strat,
            run_mode=RunMode.REMOTE if ray_kwargs else RunMode.LOCAL,
            resolution=resolution,
            engine_context=self._context,
            is_top_component=True,
        )

        self._strategies[strat] = strategy
        self._logger.debug(f"added '{strat}'")
        return strategy

    def get_strategy(self, name: str) -> BaseStrategy | ActorProxy[BaseStrategy]:
        return self._strategies[name]

    def _gather(self):
        datas: list[BaseData] = []

        for strategy in self._strategies.values():
            strategy: BaseStrategy | ActorProxy[BaseStrategy]
            strategy._gather()

            datas.extend(strategy.get_datas())
            for component in strategy.get_components():
                datas.extend(component.get_datas())

            for account in strategy.get_accounts():
                self._add_account(account)

        for data in set(datas):
            if data.category == DataCategory.MARKET_DATA:
                market_data = cast("MarketData", data)
                self._add_product(market_data.product)
            else:
                raise NotImplementedError(f"Unhandled data type: {type(data)}")

    def run(self, ctx: BaseContext | None = None):
        if ctx is not None:
            if ctx.env != self.env:
                raise ValueError(
                    f"mtflow's env {ctx.env} does not match with engine env {self.env}"
                )
            self._context.set_project_name(ctx.run.project)
            self._context.set_run_name(ctx.run.id)
        self._logger.debug(f"Running {self.name}...")
        cprint(
            f"{self.env} {self.name} is running (data_range=({self._context.data_start}, {self._context.data_end}))",
            style=self.env._color,
        )
        self._is_running = True
        self._gather()
        for venue in self._venues.values():
            venue.start()
        for strategy in self._strategies.values():
            strategy.start()

    def end(self):
        from pfeed.utils.ray import shutdown_ray

        self._logger.debug(f"Ending {self.name}...")
        for strategy in self._strategies.values():
            strategy.stop()
        for venue in self._venues.values():
            venue.stop()
        self._is_running = False
        shutdown_ray()
