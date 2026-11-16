from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    NotRequired,
    Required,
    TypedDict,
    cast,
)

if TYPE_CHECKING:
    from mtflow.tracking.run import MTFlowRun
    from pfund_kit.logging.loggers import ColoredLogger

    from pfund.components.actor_proxy import ActorProxy
    from pfund.components.strategies.strategy_base import BaseStrategy
    from pfund.engines.settings.base_engine_settings import BaseEngineSettings
    from pfund.typing import ComponentName, StrategyT

    class DataRangeDict(TypedDict, total=False):
        start_date: Required[str]
        end_date: NotRequired[str]


import logging

from pfund_blocks.enums.env import Environment
from pfund_kit.style import RichColor, TextStyle

from pfund.engines.contexts.base_engine_context import BaseEngineContext
from pfund.engines.engine_meta import EngineMeta
from pfund.enums import ComponentType, RunMode

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
        from pfund.config import setup_logging

        self._context = self.Context(**kwargs)
        setup_logging(env=self.env, config=self._context.config)
        self._logger: ColoredLogger = cast("ColoredLogger", logging.getLogger("pfund"))
        self._is_running = False
        self._owns_ray = False
        self._strategies: dict[
            ComponentName, BaseStrategy | ActorProxy[BaseStrategy]
        ] = {}

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
    def strategies(
        self,
    ) -> dict[ComponentName, BaseStrategy | ActorProxy[BaseStrategy]]:
        return self._strategies

    def is_running(self) -> bool:
        return self._is_running

    def add_strategy(
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

    def run(self, *, overwrite: bool = True, run: MTFlowRun | None = None):
        if self.is_running():
            raise RuntimeError("Engine is already running")
        # try:
        #     import mtflow
        # except ImportError:
        #     mtflow = None
        # TODO: re-enable mtflow
        mtflow = None
        if overwrite:
            if mtflow:
                self._logger.warning(
                    f"{overwrite=} is ignored when an mtflow run is active "
                    + "(every mtflow run gets its own folder, so there is nothing to overwrite)"
                )
            else:
                # TODO: moved to base_data_handler.py
                # self._clear_run_path(confirm=self.settings.warn_overwrite)
                pass
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
        from pfund_kit.utils.ray import is_ray_initialized

        # Ray already running means someone else owns it, e.g. this engine
        # runs inside a Ray actor; shutting it down would kill that actor.
        self._owns_ray = not is_ray_initialized()
        self._is_running = True
        self._setup()
        for strategy in self._strategies.values():
            strategy.start()

    def end(self, reason: str = ""):
        if not self.is_running():
            raise RuntimeError("Engine is not running")
        self._logger.warning(f"{self.env} {self.name} is ending...({reason=})")
        for strategy in self._strategies.values():
            strategy.stop()
        self._is_running = False
        self._teardown()

    def _setup(self):
        for strategy in self._strategies.values():
            strategy: BaseStrategy | ActorProxy[BaseStrategy]
            strategy._gather()

    def _teardown(self):
        from pfund_kit.utils.ray import shutdown_ray

        if self._owns_ray:
            shutdown_ray()
