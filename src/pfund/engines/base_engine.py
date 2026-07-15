# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportArgumentType=false
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    cast,
    TypeVar,
    Generic,
    TypedDict,
    ClassVar,
    Required,
    NotRequired,
)

if TYPE_CHECKING:
    from mtflow.contexts.base_context import BaseContext
    from pfund_kit.logging.loggers import ColoredLogger
    from pfeed.storages.storage_config import StorageConfig
    from pfeed.utils.file_path import FilePath

    from pfund.components.actor_proxy import ActorProxy
    from pfund.components.strategies.strategy_base import BaseStrategy
    from pfund.datas.resolution import Resolution
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
    _context: ContextT  # pyright: ignore[reportUninitializedInstanceVariable]

    def __init__(self, *, env: Environment, name: str):
        from pfund.config import setup_logging

        setup_logging(env=env)
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
        return self._context

    @property
    def run_mode(self) -> RunMode:
        return self._context.run_mode

    @property
    def settings(self) -> SettingsT:
        return self._context.settings

    def is_running(self) -> bool:
        return self._is_running

    def _create_context(
        self,
        *,
        env: Environment,
        name: str,
        data_range: str | Resolution | DataRangeDict | tuple[str, str] | Literal["ytd"],
        settings: SettingsT | None = None,
        storage_config: StorageConfig | None = None,
        **kwargs: Any,
    ) -> ContextT:
        return cast(
            "ContextT",
            self.Context(
                env=env,
                name=name,
                data_range=data_range,
                settings=settings,
                storage_config=storage_config,
                **kwargs,
            ),
        )

    def add_strategy(
        self,
        strategy: StrategyT,
        resolution: str,
        name: str = "",
        storage_config: StorageConfig | None = None,
        ray_actor_options: dict[str, Any] | None = None,
        **ray_kwargs: Any,
    ) -> StrategyT | ActorProxy[StrategyT]:
        """
        Args:
            storage_config:
                per-strategy override for where this strategy's artifacts are persisted.
                Falls back to the engine-level default (self._storage_config) when None.
            df_form: DataFrame layout used by this strategy.
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
            storage_config=storage_config or self.context.storage_config,
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
        run_id: str,
    ) -> FilePath:
        return data_path / "runs" / f"env={env}" / project_name.lower() / run_id.lower()

    def _clear_run_path(self, *, confirm: bool = False) -> None:
        import pyarrow.fs as pa_fs

        from pfeed.storages.file_based_storage import FileBasedStorage
        from pfeed.utils.file_path import FilePath
        from pfeed.enums import DataStorage

        storage_config = self.context.storage_config
        Storage = DataStorage[storage_config.storage].storage_class
        storage = Storage.from_storage_config(storage_config)

        if not isinstance(storage, FileBasedStorage):
            raise TypeError(f"Cannot clear a filesystem run path using {storage.name}")

        run_path = FilePath(
            self._create_run_path(
                data_path=storage.data_path,
                env=self.env,
                project_name=self.context.project_name,
                run_id=self.context.run_id,
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

        class _NewRunCancelled(Exception):
            pass

        def _handle_sigint(_signum, _frame):
            raise _NewRunCancelled

        previous_sigint_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, _handle_sigint)
        try:
            choice = Prompt.ask(
                (
                    "\n[bold red]WARNING:[/] engine's run(new=True) will permanently delete "
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
        except _NewRunCancelled:
            choice = "n"
        finally:
            signal.signal(signal.SIGINT, previous_sigint_handler)

        if choice == "n":
            print("\nRun cancelled; the existing run folder was not cleared")
            raise SystemExit(0)

        if choice == "d":
            self.settings.warn_new_run = False
            self.context._save_settings(self.settings)

    def run(self, ctx: BaseContext | None = None, new: bool = True):
        if ctx is not None:
            if ctx.env != self.env:
                raise ValueError(
                    f"mtflow's env {ctx.env} does not match with engine env {self.env}"
                )
            self._context.set_project_name(ctx.run.project)
            self._context.set_run_id(ctx.run.id)
        else:
            if new:
                self._clear_run_path(confirm=self.settings.warn_new_run)
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
