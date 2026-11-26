from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
    NotRequired,
    Required,
    TypedDict,
    cast,
    overload,
)

if TYPE_CHECKING:
    from mtflow.tracking.run import MTFlowRun
    from pfund_blocks.entities import BaseAccount, BaseProduct, BaseVenue
    from pfund_blocks.entities.venues.venue_base import VenueConfig
    from pfund_blocks.typing import TradingVenue
    from pfund_kit.logging.loggers import ColoredLogger

    from pfund.components.actor_proxy import ActorProxy
    from pfund.components.strategies.strategy_base import BaseStrategy
    from pfund.engines.settings.base_engine_settings import BaseEngineSettings
    from pfund.typing import ComponentName

    class DataRangeDict(TypedDict, total=False):
        start_date: Required[str]
        end_date: NotRequired[str]


import logging

from pfund_blocks.enums.env import Environment
from pfund_kit.style import RichColor, TextStyle

from pfund.engines.contexts.base_engine_context import BaseEngineContext
from pfund.engines.engine_meta import EngineMeta
from pfund.enums import ComponentType, RunMode
from pfund.managers import OrderManager, PortfolioManager, RiskManager

_ENV_COLORS: dict[Environment, str] = {
    Environment.BACKTEST: TextStyle.BOLD + RichColor.BLUE,
    Environment.SANDBOX: TextStyle.BOLD + RichColor.GREY0,
    Environment.PAPER: TextStyle.BOLD + RichColor.RED,
    Environment.LIVE: TextStyle.BOLD + RichColor.GREEN,
}


class BaseEngine[
    SettingsT: BaseEngineSettings,
    ContextT: BaseEngineContext[Any],
](metaclass=EngineMeta):
    Context: ClassVar[type[BaseEngineContext[Any]]] = BaseEngineContext

    def __init__(self, **kwargs: Any):
        import pfeed as pe

        from pfund.config import setup_logging

        self._context = self.Context(**kwargs)
        setup_logging(env=self.env, config=self._context.config)
        self._logger: ColoredLogger = cast("ColoredLogger", logging.getLogger("pfund"))
        self._is_running = False
        self._is_built = False
        self._strategies: dict[
            ComponentName, BaseStrategy | ActorProxy[BaseStrategy]
        ] = {}
        self._feed = pe.PFund().engine_feed
        self._venues: dict[TradingVenue, BaseVenue] = {}
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
    def context(self) -> ContextT:
        return cast("ContextT", self._context)

    @property
    def run_mode(self) -> RunMode:
        return self._context.run_mode

    @property
    def settings(self) -> SettingsT:
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

    @property
    def strategies(
        self,
    ) -> dict[ComponentName, BaseStrategy | ActorProxy[BaseStrategy]]:
        return self._strategies

    def is_running(self) -> bool:
        return self._is_running

    def _add_product(self, product: BaseProduct):
        for _venue in self._venues.values():
            if (existing := _venue.products.get(product.name)) is not None:
                raise ValueError(
                    f'product name "{product.name}" is already used by {existing!r}; '
                    + "product names must be unique across the engine"
                )
        venue: BaseVenue = self.add_venue(product.source)
        venue.add_product(product)

    def _add_account(self, account: BaseAccount):
        if self.env.is_simulated():
            from pfund.simulated.simulated_account import SimulatedAccount

            if not isinstance(account, SimulatedAccount):
                raise TypeError(
                    f"env={self.env} only accepts pf.SimulatedAccount, got {account.__class__.__name__}, "
                    + f"e.g. pf.SimulatedAccount(venue={account.venue!r}, balances=..., positions=...)"
                )

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
        venue: BaseVenue = self.add_venue(account.venue)
        venue.add_account(account)

    # TODO
    def _create_venue(
        self, venue: TradingVenue, config: VenueConfig | None = None
    ) -> BaseVenue:
        """Creates the venue the engine plugs in, the ONLY place that decides real vs simulated venue by env.

        env=BACKTEST/SANDBOX plugs in the simulated venue (the fake exchange), env=PAPER/LIVE plugs in the real venue.
        """
        if self.env.is_simulated():
            from pfund.simulated.simulated_venue import SimulatedVenue

            # TODO: pass self._simulated_venue_configs.get(venue) and config (for the internal read-only real venue)
            # when SimulatedVenue accepts them
            return SimulatedVenue.for_venue(venue)(
                env=cast("Literal[Environment.BACKTEST, Environment.SANDBOX]", self.env),
                engine_feed=self._feed,
                config=config,
            )
        else:
            from pfund.registry import get_venue

            Venue = get_venue(venue)
            # TODO: pass env and config when BaseVenue's __init__ accepts them
            return Venue(env=self.env, config=config, read_only=False)

    def add_venue(
        self, venue: TradingVenue, config: VenueConfig | None = None
    ) -> BaseVenue:
        venue = venue.upper()
        if venue not in self._venues:
            self._venues[venue] = self._create_venue(venue, config=config)
            self._logger.debug(f"added trading venue {venue}")
        elif config is not None:
            raise ValueError(f"{venue} already exists and cannot be configured")
        return self._venues[venue]

    def get_venue(self, venue: TradingVenue) -> BaseVenue:
        return self._venues[venue.upper()]

    # local strategy in + num_cpus -> wrapped into a remote strategy
    @overload
    def add_strategy[StrategyT: BaseStrategy](
        self,
        strategy: StrategyT,
        resolution: str,
        name: str = "",
        ray_actor_options: dict[str, Any] | None = None,
        *,
        num_cpus: float,
        **ray_kwargs: Any,
    ) -> ActorProxy[StrategyT]: ...

    # local strategy in, no num_cpus -> local strategy out
    @overload
    def add_strategy[StrategyT: BaseStrategy](
        self,
        strategy: StrategyT,
        resolution: str,
        name: str = "",
        ray_actor_options: dict[str, Any] | None = None,
        *,
        num_cpus: None = None,
        **ray_kwargs: Any,
    ) -> StrategyT: ...

    def add_strategy[StrategyT: BaseStrategy](
        self,
        strategy: StrategyT,
        resolution: str,
        name: str = "",
        ray_actor_options: dict[str, Any] | None = None,
        **ray_kwargs: Any,
    ) -> StrategyT | ActorProxy[StrategyT]:
        """Add a strategy to the engine.

        Args:
            strategy: Strategy instance to add.
            resolution: Resolution at which the strategy runs.
            name: Optional name for the strategy.
            ray_actor_options: Options passed to the Ray actor.
            ray_kwargs: Ray actor constructor arguments. Providing these runs the
                strategy remotely.

        Returns:
            The added strategy or its remote proxy.
        """
        from pfund.components.actor_proxy import ActorProxy
        from pfund.components.strategies.strategy_base import BaseStrategy

        Strategy = strategy.__class__
        StrategyName = Strategy.__name__
        assert isinstance(strategy, BaseStrategy), (
            f"strategy '{StrategyName}' is not an instance of BaseStrategy. Please create your strategy using 'class {StrategyName}(pf.Strategy)'"
        )

        strat = name or strategy.name
        if any(
            strat.casefold() == existing.casefold() for existing in self._strategies
        ):
            raise ValueError(f"{strat} already exists")

        if ray_kwargs:
            strategy: ActorProxy[StrategyT] = ActorProxy(
                strategy,
                name=strat,
                resolution=resolution,
                df_form="long",
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
            df_form="long",
        )

        # ActorProxy is invariant in its component type, so widen explicitly
        self._strategies[strat] = cast(
            "BaseStrategy | ActorProxy[BaseStrategy]", strategy
        )
        self._logger.debug(f"added '{strat}'")
        return strategy

    def get_strategy(self, name: str) -> BaseStrategy | ActorProxy[BaseStrategy]:
        return self._strategies[name]

    def run(self, *, run: MTFlowRun | None = None):
        if self.is_running():
            raise RuntimeError("Engine is already running")
        # try:
        #     import mtflow
        # except ImportError:
        #     mtflow = None
        # TODO: re-enable mtflow
        mtflow = None
        run = run or (mtflow.get_run() if mtflow else None)
        if run is not None:
            assert mtflow is not None
            client = mtflow.get_client()
            assert client is not None
            mtflow.configure(
                # NOTE: use datalake_path, not pfund's data_path
                data_path=str(self._context.datalake_storage_config.data_path)
            )
            client.set_env(self.env)
            # TODO: removed these 2 functions, should use data handler
            # self._context.set_project_name(run.project)
            # self._context.set_run_name(run.name)
        self._logger.warning(
            f"{self.env} {self.name} is running (data_range=({self._context.data_start}, {self._context.data_end}))",
            style=_ENV_COLORS[self.env],
        )
        if not self._is_built:
            self.build()
        self._is_running = True

        for strategy in self._strategies.values():
            strategy.start()

    def end(self, reason: str = ""):
        self._logger.warning(f"{self.env} {self.name} is ending...({reason=})")
        for strategy in self._strategies.values():
            strategy.stop()
        self._is_running = False

    def _assert_unique_component_names(self):
        """Rejects two components with the same name anywhere in the engine.

        Each parent only rejects duplicates among its own children, but ZMQ
        ports and urls in the engine settings are keyed by bare component name,
        so e.g. two strategies each with a model named "model_x" would clash.
        """
        names = [
            name.casefold()
            for strat, strategy in self._strategies.items()
            for name in (strat, *strategy._get_descendants())
        ]
        if duplicates := sorted({name for name in names if names.count(name) > 1}):
            raise ValueError(
                f"Component names must be unique across the engine, duplicates: {duplicates}"
            )

    def build(self) -> None:
        """Gathers and validates every component, then loads their data and builds their stores (e.g. X).

        After build(), components are ready to use, e.g. in vectorized backtesting:

            engine.build()
            engine.use_split("train")
            strategy.backtest(strategy.X)

        run() calls it if it has not been called. Calling it again is a no-op.
        It does not start the engine or the strategies.
        """
        if self._is_built:
            self._logger.debug(f"{self.name} has already built")
            return
        # NOTE: gather the whole graph first, so it is validated before any data is loaded
        for strategy in self._strategies.values():
            strategy: BaseStrategy | ActorProxy[BaseStrategy]
            strategy._gather()
        self._assert_unique_component_names()
        # TODO: mtflow, build() saves the source artifacts before run(run=...) could configure a tracking run
        for strategy in self._strategies.values():
            strategy._build()
        self._is_built = True
