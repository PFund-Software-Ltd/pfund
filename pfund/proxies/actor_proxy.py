from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ray.actor import ActorHandle
    from pfund.typing import Component

from pfund import cprint


class ActorProxy:
    def __init__(self, component: Component, name: str='', ray_actor_options: dict | None=None, **ray_kwargs):
        import ray
        Component = component.__class__
        component_name = name or component.name
        try:
            ComponentActor = ray.remote(**ray_kwargs)(Component)
        except ValueError as err:
            raise ValueError(f"{Component.__name__} {ray_kwargs=}:\n{err}")
        ray_actor_options = ray_actor_options or {}
        if 'name' not in ray_actor_options:
            ray_actor_options['name'] = component_name
        self._actor: ActorHandle = (
            ComponentActor
            .options(**ray_actor_options)
            .remote(*component.__pfund_args__, **component.__pfund_kwargs__)
        )
        cprint(f'Ray Actor "{component_name}" is created', style='bold')

        # set up essential attributes for convenience
        self.name = component_name
        self.resolution = component.resolution
        
    # NOTE: added __setstate__ and __getstate__ to avoid ray's serialization issues when returning ActorProxy objects
    def __setstate__(self, state):
        self.__dict__.update(state)
        
    def __getstate__(self):
        return self.__dict__
    
    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        else:
            import ray
            actor = self.__dict__["_actor"]
            attr = getattr(actor, name)
            def remote_method(*args, **kwargs):
                return ray.get(attr.remote(*args, **kwargs))
            return remote_method
