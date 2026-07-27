# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportArgumentType=false
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    cast,
    TypeVar,
    Generic,
    TypedDict,
    ClassVar,
    Required,
    NotRequired,
)


if TYPE_CHECKING:
    from mtflow.tracking.run import MTFlowRun
    from pfund_kit.logging.loggers import ColoredLogger
    from pfeed.utils.file_path import FilePath

    from pfund.components.actor_proxy import ActorProxy
    from pfund.components.strategies.strategy_base import BaseStrategy
    from pfund.typing import StrategyT, ComponentName

    class DataRangeDict(TypedDict, total=False):
        start_date: Required[str]
        end_date: NotRequired[str]


import logging

from pfund_kit.utils.singleton import SingletonMeta

from pfund.enums import ComponentType, Environment, RunMode
from pfund.engines.contexts.base_engine_context import BaseEngineContext, SettingsT


ContextT = TypeVar("ContextT", bound="BaseEngineContext[Any]")


class BaseEngine(Generic[SettingsT, ContextT], metaclass=SingletonMeta):
    Context: ClassVar[type[BaseEngineContext[Any]]] = BaseEngineContext

    def __init__(self, **kwargs: Any):
        from pfund.config import setup_logging

        self._context = self.Context(**kwargs)
        setup_logging(env=self.env)
        self._logger: ColoredLogger = cast("ColoredLogger", logging.getLogger("pfund"))
        self._is_running = False
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
            df_form="long",
        )

        self._strategies[strat] = strategy
        self._logger.debug(f"added '{strat}'")
        return strategy

    def get_strategy(self, name: str) -> BaseStrategy | ActorProxy[BaseStrategy]:
        return self._strategies[name]

    @staticmethod
    def _create_run_path(
        *,
        data_path: FilePath,
        env: Environment,
        project_name: str,
        run_name: str,
    ) -> FilePath:
        return (
            data_path / "runs" / f"env={env}" / project_name.lower() / run_name.lower()
        )

    def _clear_run_path(self, *, confirm: bool = False) -> None:
        import pyarrow.fs as pa_fs

        from pfeed.storages.file_based_storage import FileBasedStorage
        from pfeed.utils.file_path import FilePath
        from pfeed.enums import DataStorage

        storage_config = self.context.datalake_storage_config
        Storage = DataStorage[storage_config.storage].storage_class
        storage = Storage.from_storage_config(storage_config)

        if not isinstance(storage, FileBasedStorage):
            raise TypeError(f"Cannot clear a filesystem run path using {storage.name}")

        run_path = FilePath(
            self._create_run_path(
                data_path=storage.data_path,
                env=self.env,
                project_name=self.context.project_name,
                run_name=self.context.run_name,
            )
        )

        filesystem = storage.get_filesystem()
        file_info = filesystem.get_file_info(run_path.schemeless)

        if file_info.type == pa_fs.FileType.NotFound:
            return

        if file_info.type != pa_fs.FileType.Directory:
            raise RuntimeError(f"Run path is not a directory: {run_path}")

        if confirm:
            self._confirm_clear_run_path(run_path)

        filesystem.delete_dir(run_path.schemeless)
        self._logger.debug(f"cleared existing run path: {run_path}")

    def _confirm_clear_run_path(self, run_path: FilePath) -> None:
        import signal

        from rich.markup import escape
        from rich.prompt import Prompt

        class _OverwriteCancelled(Exception):
            pass

        def _handle_sigint(_signum, _frame):
            raise _OverwriteCancelled

        previous_sigint_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, _handle_sigint)
        try:
            choice = Prompt.ask(
                (
                    "\n[bold red]WARNING:[/] engine's run(overwrite=True) will permanently delete "
                    "the whole run folder:\n\n"
                    f"  {escape(str(run_path))}\n\n"
                    "This includes all component artifacts (e.g. models).\n\n"
                    "\\[y] Clear this time\n"
                    "\\[n] Cancel\n"
                    "\\[d] Clear and disable this warning in settings.toml"
                ),
                choices=["y", "n", "d"],
                case_sensitive=False,
                default="n",
                show_choices=False,
            ).lower()
        except _OverwriteCancelled:
            choice = "n"
        finally:
            signal.signal(signal.SIGINT, previous_sigint_handler)

        if choice == "n":
            print("\nRun cancelled; the existing run folder was not cleared")
            raise SystemExit(0)

        if choice == "d":
            self.settings.warn_overwrite = False
            self.context._save_settings(self.settings)

    def run(self, *, overwrite: bool = True, run: MTFlowRun | None = None):
        try:
            import mtflow
        except ImportError:
            mtflow = None
        if overwrite:
            if mtflow:
                self._logger.warning(
                    f"{overwrite=} is ignored when an mtflow run is active "
                    + "(every mtflow run gets its own folder, so there is nothing to overwrite)"
                )
            else:
                self._clear_run_path(confirm=self.settings.warn_overwrite)
        run = run or (mtflow.get_run() if mtflow else None)
        if run is not None:
            assert mtflow is not None
            client = mtflow.get_client()
            assert client is not None
            client.set_env(self.env)
            self._context.set_project_name(run.project)
            self._context.set_run_name(run.name)
        self._logger.warning(
            f"{self.env} {self.name} is running (data_range=({self._context.data_start}, {self._context.data_end}))",
            style=self.env._color,
        )
        self._is_running = True
        self._setup()
        for strategy in self._strategies.values():
            strategy.start()

    def end(self):
        self._logger.warning(f"{self.env} {self.name} is ending...")
        for strategy in self._strategies.values():
            strategy.stop()
        self._is_running = False
        self._teardown()

    def _setup(self):
        for strategy in self._strategies.values():
            strategy: BaseStrategy | ActorProxy[BaseStrategy]
            strategy._gather()

    def _teardown(self):
        from pfeed.utils.ray import shutdown_ray

        shutdown_ray()
