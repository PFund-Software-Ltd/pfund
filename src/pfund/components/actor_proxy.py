from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Self, cast, overload

if TYPE_CHECKING:
    from pathlib import Path

    from ray.actor import ActorClass, ActorHandle

    from pfund.components.features.feature_base import BaseFeature
    from pfund.components.models.model_base import BaseModel, UnderlyingModel
    from pfund.components.strategies.strategy_base import BaseStrategy
    from pfund.engines.contexts.base_engine_context import BaseEngineContext
    from pfund.typing import Component

from pfund_blocks.resolution import Resolution

from pfund.enums import ComponentType


class ActorProxy[ComponentT: Component]:
    def __init__(
        self,
        component: ComponentT,
        name: str,
        resolution: Resolution | str,
        df_form: Literal["wide", "long"],
        component_type: ComponentType,
        engine_context: BaseEngineContext[Any],
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
        self.df_form: Literal["wide", "long"] = df_form
        self.component_type: ComponentType = component_type
        self.context: BaseEngineContext[Any] = engine_context
        if isinstance(self.context.settings, TradeEngineSettings):
            self.context.settings.zmq_urls.enable_ray()
            self.context.settings.zmq_ports.enable_ray()

    def _ensure_actor(self) -> ActorHandle[ComponentT]:
        if self._actor is None:
            import ray
            from pfund_kit.utils.ray import setup_ray

            assert self._hydrate_fields is not None, "ActorProxy is not hydrated"
            setup_ray()
            self._actor = self._create_actor(
                self._component, self._ray_actor_options, **self._ray_kwargs
            )
            # NOTE: ray types ActorHandle.__getattr__ as Never, which makes the
            # rest unreachable for type checkers, getattr() returns Any instead
            ray.get(getattr(self._actor, "_hydrate").remote(**self._hydrate_fields))  # noqa: B009
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
                ComponentActor.options(**ray_actor_options).remote(
                    *component.__pfund_args__, **component.__pfund_kwargs__
                )
            ),
        )
        ray.get(
            # NOTE: getattr() for the same reason as in _ensure_actor()
            getattr(actor, "_set_source_artifact_path").remote(  # noqa: B009
                str(source_artifact_path)
            )
        )
        # NOTE: the actor rebuilds the component from its constructor args only,
        # so config/params loaded on the local instance (e.g. via load_params()) must be pushed explicitly
        ray.get(
            # NOTE: getattr() for the same reason as in _ensure_actor()
            getattr(actor, "_set_config_and_params").remote(  # noqa: B009
                component.config, component.params
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

    def load_config(
        self, config: dict[str, Any] | None = None, file_path: str | Path = ""
    ) -> Self:
        """Loads config into the remote component, see ComponentBaseMixin.load_config.

        Returns:
            The proxy itself (not the remote component), so the call can be chained.
        """
        return self._load("config", config, file_path)

    def load_params(
        self, params: dict[str, Any] | None = None, file_path: str | Path = ""
    ) -> Self:
        """Loads params into the remote component, see ComponentBaseMixin.load_params.

        Returns:
            The proxy itself (not the remote component), so the call can be chained.
        """
        return self._load("params", params, file_path)

    def _load(
        self,
        kind: Literal["config", "params"],
        values: dict[str, Any] | None,
        file_path: str | Path,
    ) -> Self:
        if self._actor is None:
            # actor not created yet, load into the local instance,
            # _create_actor() pushes its config/params to the actor on creation
            getattr(self._component, f"load_{kind}")(values, file_path)
        else:
            import ray

            # read the file here, in the caller's process, so the actor never needs the file
            # (it may run on another node), then only the dict is sent over
            values = self._component._load_dict_or_toml(kind, values, file_path)
            # NOTE: not forwarded via __getattr__: load_*() returns the component itself,
            # which Ray would serialize and copy back here
            ray.get(
                # NOTE: getattr() for the same reason as in _ensure_actor()
                getattr(self._actor, "_load_via_proxy").remote(kind, values)  # noqa: B009
            )
        return self

    def _add_component(
        self,
        add_method: Literal["add_strategy", "add_model", "add_feature"],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        import ray

        actor = self._ensure_actor()
        # NOTE: getattr() for the same reason as in _ensure_actor()
        return ray.get(
            getattr(actor, "_add_component_via_proxy").remote(  # noqa: B009
                add_method, *args, **kwargs
            )
        )

    # remote strategy in -> remote strategy out, ray args are ignored (the proxy is already remote)
    @overload
    def add_strategy[StrategyT: BaseStrategy](
        self,
        strategy: ActorProxy[StrategyT],
        resolution: str = "",
        name: str = "",
        ray_actor_options: dict[str, Any] | None = None,
        **ray_kwargs: Any,
    ) -> ActorProxy[StrategyT]: ...

    # local strategy in + num_cpus -> wrapped into a remote strategy
    @overload
    def add_strategy[StrategyT: BaseStrategy](
        self,
        strategy: StrategyT,
        resolution: str = "",
        name: str = "",
        ray_actor_options: dict[str, Any] | None = None,
        *,
        num_cpus: float,
        **ray_kwargs: Any,
    ) -> ActorProxy[StrategyT]: ...

    # local strategy in, no num_cpus -> None, the strategy lives in this proxy's actor
    @overload
    def add_strategy[StrategyT: BaseStrategy](
        self,
        strategy: StrategyT,
        resolution: str = "",
        name: str = "",
        ray_actor_options: dict[str, Any] | None = None,
        *,
        num_cpus: None = None,
        **ray_kwargs: Any,
    ) -> None: ...

    def add_strategy[StrategyT: BaseStrategy](
        self,
        strategy: StrategyT | ActorProxy[StrategyT],
        resolution: str = "",
        name: str = "",
        ray_actor_options: dict[str, Any] | None = None,
        **ray_kwargs: Any,
    ) -> ActorProxy[StrategyT] | None:
        """Add a child strategy to the remote strategy, see BaseStrategy.add_strategy.

        Returns:
            The child's remote proxy if the child runs remotely, otherwise ``None``,
            since a local child lives in this proxy's actor and can't be returned
            without being copied.
        """
        return self._add_component(
            "add_strategy",
            strategy=strategy,
            resolution=resolution,
            name=name,
            ray_actor_options=ray_actor_options,
            **ray_kwargs,
        )

    # remote model in -> remote model out, ray args are ignored (the proxy is already remote)
    @overload
    def add_model[ModelT: BaseModel](
        self,
        model: ActorProxy[ModelT],
        resolution: str = "",
        name: str = "",
        df_form: Literal["wide", "long"] = "wide",
        ray_actor_options: dict[str, Any] | None = None,
        **ray_kwargs: Any,
    ) -> ActorProxy[ModelT]: ...

    # local model in + num_cpus -> wrapped into a remote model
    @overload
    def add_model[ModelT: BaseModel](
        self,
        model: ModelT,
        resolution: str = "",
        name: str = "",
        df_form: Literal["wide", "long"] = "wide",
        ray_actor_options: dict[str, Any] | None = None,
        *,
        num_cpus: float,
        **ray_kwargs: Any,
    ) -> ActorProxy[ModelT]: ...

    # local model in, no num_cpus -> None, the model lives in this proxy's actor
    @overload
    def add_model[ModelT: BaseModel](
        self,
        model: ModelT,
        resolution: str = "",
        name: str = "",
        df_form: Literal["wide", "long"] = "wide",
        ray_actor_options: dict[str, Any] | None = None,
        *,
        num_cpus: None = None,
        **ray_kwargs: Any,
    ) -> None: ...

    # underlying model (e.g. sklearn/pytorch) in + num_cpus -> wrapped pfund model as a remote model
    @overload
    def add_model(
        self,
        model: UnderlyingModel,
        resolution: str = "",
        name: str = "",
        df_form: Literal["wide", "long"] = "wide",
        ray_actor_options: dict[str, Any] | None = None,
        *,
        num_cpus: float,
        **ray_kwargs: Any,
    ) -> ActorProxy[BaseModel]: ...

    # underlying model in, no num_cpus -> None, the wrapped model lives in this proxy's actor
    @overload
    def add_model(
        self,
        model: UnderlyingModel,
        resolution: str = "",
        name: str = "",
        df_form: Literal["wide", "long"] = "wide",
        ray_actor_options: dict[str, Any] | None = None,
        *,
        num_cpus: None = None,
        **ray_kwargs: Any,
    ) -> None: ...

    def add_model[ModelT: BaseModel](
        self,
        model: ModelT | ActorProxy[ModelT] | UnderlyingModel,
        resolution: str = "",
        name: str = "",
        df_form: Literal["wide", "long"] = "wide",
        ray_actor_options: dict[str, Any] | None = None,
        **ray_kwargs: Any,
    ) -> ActorProxy[ModelT] | ActorProxy[BaseModel] | None:
        """Add a model as a child of the remote component, see ComponentMixin.add_model.

        Returns:
            The child's remote proxy if the child runs remotely, otherwise ``None``,
            since a local child lives in this proxy's actor and can't be returned
            without being copied.
        """
        return self._add_component(
            "add_model",
            model=model,
            resolution=resolution,
            name=name,
            df_form=df_form,
            ray_actor_options=ray_actor_options,
            **ray_kwargs,
        )

    # remote feature in -> remote feature out, ray args are ignored (the proxy is already remote)
    @overload
    def add_feature[FeatureT: BaseFeature](
        self,
        feature: ActorProxy[FeatureT],
        resolution: str = "",
        name: str = "",
        df_form: Literal["wide", "long"] = "wide",
        ray_actor_options: dict[str, Any] | None = None,
        **ray_kwargs: Any,
    ) -> ActorProxy[FeatureT]: ...

    # local feature in + num_cpus -> wrapped into a remote feature
    @overload
    def add_feature[FeatureT: BaseFeature](
        self,
        feature: FeatureT,
        resolution: str = "",
        name: str = "",
        df_form: Literal["wide", "long"] = "wide",
        ray_actor_options: dict[str, Any] | None = None,
        *,
        num_cpus: float,
        **ray_kwargs: Any,
    ) -> ActorProxy[FeatureT]: ...

    # local feature in, no num_cpus -> None, the feature lives in this proxy's actor
    @overload
    def add_feature[FeatureT: BaseFeature](
        self,
        feature: FeatureT,
        resolution: str = "",
        name: str = "",
        df_form: Literal["wide", "long"] = "wide",
        ray_actor_options: dict[str, Any] | None = None,
        *,
        num_cpus: None = None,
        **ray_kwargs: Any,
    ) -> None: ...

    def add_feature[FeatureT: BaseFeature](
        self,
        feature: FeatureT | ActorProxy[FeatureT],
        resolution: str = "",
        name: str = "",
        df_form: Literal["wide", "long"] = "wide",
        ray_actor_options: dict[str, Any] | None = None,
        **ray_kwargs: Any,
    ) -> ActorProxy[FeatureT] | None:
        """Add a feature as a child of the remote component, see ComponentMixin.add_feature.

        Returns:
            The child's remote proxy if the child runs remotely, otherwise ``None``,
            since a local child lives in this proxy's actor and can't be returned
            without being copied.
        """
        return self._add_component(
            "add_feature",
            feature=feature,
            resolution=resolution,
            name=name,
            df_form=df_form,
            ray_actor_options=ray_actor_options,
            **ray_kwargs,
        )

    def __getattr__(self, name: str) -> Any:
        if name in self.__dict__:
            return self.__dict__[name]

        import ray

        actor = self._ensure_actor()
        attr = getattr(actor, name)

        def remote_method(*args: Any, **kwargs: Any) -> Any:
            return ray.get(attr.remote(*args, **kwargs))

        return remote_method
