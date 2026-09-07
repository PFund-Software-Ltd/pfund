from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, cast

if TYPE_CHECKING:
    from ray.actor import ActorClass, ActorHandle

    from pfund.engines.contexts.trade_engine_context import TradeEngineContext

from pfund.datas.resolution import Resolution
from pfund.enums import ComponentType
from pfund.typing import ComponentT


class ActorProxy(Generic[ComponentT]):
    def __init__(
        self,
        component: ComponentT,
        name: str,
        resolution: Resolution | str,
        component_type: ComponentType,
        engine_context: TradeEngineContext,
        ray_actor_options: dict[str, Any] | None = None,
        **ray_kwargs: Any,
    ):
        from pfund.engines.settings.trade_engine_settings import TradeEngineSettings

        if "num_cpus" not in ray_kwargs:
            raise ValueError("`num_cpus` must be set for a Ray actor")
        if ray_kwargs["num_cpus"] <= 0:
            raise ValueError("`num_cpus` must be greater than 0")
        ray_actor_options = ray_actor_options or {}
        ray_actor_options.setdefault("name", name)

        # Adding a component must not cost a process: the actor is created on
        # first use, so a component tree can be built or loaded without running it.
        self._component = component
        self._ray_actor_options = ray_actor_options
        self._ray_kwargs = ray_kwargs
        self._actor: ActorHandle[ComponentT] | None = None
        self._hydrate_fields: dict[str, Any] | None = None
        self.name: str = name
        self.resolution: Resolution = Resolution(resolution)
        self.component_type: ComponentType = component_type
        self.context: TradeEngineContext = engine_context
        if isinstance(self.context.settings, TradeEngineSettings):
            self.context.settings.zmq_urls.enable_ray()
            self.context.settings.zmq_ports.enable_ray()

    def _ensure_actor(self) -> ActorHandle[ComponentT]:
        if self._actor is None:
            import ray
            from pfeed.utils.ray import setup_ray

            assert self._hydrate_fields is not None, "ActorProxy is not hydrated"
            setup_ray()
            self._actor = self._create_actor(
                self._component, self._ray_actor_options, **self._ray_kwargs
            )
            ray.get(getattr(self._actor, "_hydrate").remote(**self._hydrate_fields))
        return self._actor

    def _hydrate(self, **kwargs: Any) -> None:
        self._hydrate_fields = kwargs

    @staticmethod
    def _create_actor(
        component: ComponentT, ray_actor_options: dict[str, Any], **ray_kwargs: Any
    ) -> ActorHandle[ComponentT]:
        import ray

        source_artifact_path = component._source_artifact.resolve()
        ComponentClass: type[ComponentT] = component.__class__
        try:
            ComponentActor: ActorClass[ComponentT] = ray.remote(**ray_kwargs)(
                ComponentClass
            )
        except ValueError as err:
            raise ValueError(
                f"{ComponentClass.__name__} {ray_kwargs=}:\n{err}"
            ) from err

        actor = cast(
            "ActorHandle[ComponentT]",
            (
                ComponentActor.options(**ray_actor_options).remote(  # pyright: ignore[reportUnknownMemberType]
                    *component.__pfund_args__, **component.__pfund_kwargs__
                )
            ),
        )
        ray.get(
            getattr(actor, "_set_source_artifact_path").remote(
                str(source_artifact_path)
            )
        )
        return actor

    @property
    def actor(self) -> ActorHandle[ComponentT]:
        return self._ensure_actor()

    def __getstate__(self) -> dict[str, Any]:
        # A copy in another process must share the actor, not spawn its own.
        self._ensure_actor()
        return self.__dict__

    # NOTE: added __setstate__ and __getstate__ to avoid ray's serialization issues when returning ActorProxy objects
    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

    def __getattr__(self, name: str) -> Any:
        if name in self.__dict__:
            return self.__dict__[name]

        import ray

        actor = self._ensure_actor()
        attr = getattr(actor, name)

        def remote_method(*args: Any, **kwargs: Any) -> Any:
            try:
                return ray.get(attr.remote(*args, **kwargs))
            except TypeError as err:
                # NOTE: catch TypeError when trying to pickle and return a component
                # e.g. model = strategy.add_model(...), where strategy is a ray actor but model is not, so model can't be serialized and returned correctly
                # if 'cannot pickle' in str(err):
                #     print_error(f'Ray Actor "{self.name}" error when calling "{name}": {err}')
                #     return None
                # else:
                raise err

        return remote_method
