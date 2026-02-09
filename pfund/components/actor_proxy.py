from __future__ import annotations
from typing import TYPE_CHECKING, Any, cast, Generic
from typing_extensions import override
if TYPE_CHECKING:
    from ray.actor import ActorHandle, ActorClass
    from pfund.datas.resolution import Resolution

from pfund.typing import ComponentT
from pfund_kit.style import cprint


class ActorProxy(Generic[ComponentT]):
    def __init__(self, component: ComponentT, name: str='', ray_actor_options: dict[str, Any] | None=None, **ray_kwargs: Any):
        if 'num_cpus' not in ray_kwargs:
            raise ValueError('`num_cpus` must be set for a Ray actor')
        if ray_kwargs['num_cpus'] <= 0:
            raise ValueError('`num_cpus` must be greater than 0')
        component_name = name or component.name
        ray_actor_options = ray_actor_options or {}
        if 'name' not in ray_actor_options:
            ray_actor_options['name'] = component_name
        self._actor: ActorHandle[ComponentT] = self._create_actor(component, ray_actor_options, **ray_kwargs)
        cprint(f'Ray Actor "{component_name}" is created', style='bold')

        # set up essential attributes for convenience
        self.name: str = component_name
        self.resolution: Resolution | None = component.resolution
    
    @staticmethod
    def _create_actor(component: ComponentT, ray_actor_options: dict[str, Any], **ray_kwargs: Any) -> ActorHandle[ComponentT]:
        import ray
        ComponentClass: type[ComponentT] = component.__class__
        try:
            ComponentActor: ActorClass[ComponentT] = ray.remote(**ray_kwargs)(ComponentClass)
        except ValueError as err:
            raise ValueError(f"{ComponentClass.__name__} {ray_kwargs=}:\n{err}")
        
        return cast(
            ActorHandle[ComponentT], (
                ComponentActor  # pyright: ignore[reportUnknownMemberType]
                .options(**ray_actor_options)
                .remote(*component.__pfund_args__, **component.__pfund_kwargs__)
            )
        )
        
    # NOTE: added __setstate__ and __getstate__ to avoid ray's serialization issues when returning ActorProxy objects
    def __setstate__(self, state: dict[str, Any]):
        self.__dict__.update(state)
        
    @override
    def __getstate__(self) -> dict[str, Any]:
        return self.__dict__
    
    def __getattr__(self, name: str) -> Any:
        if name in self.__dict__:
            return self.__dict__[name]
        else:
            import ray
            actor = self.__dict__["_actor"]
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
