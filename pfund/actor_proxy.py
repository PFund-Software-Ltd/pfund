from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ray.actor import ActorHandle


class ActorProxy:
    def __init__(self, actor: ActorHandle):
        self._actor = actor

    def __getattr__(self, name):
        import ray
        attr = getattr(self._actor, name)
        def remote_method(*args, **kwargs):
            return ray.get(attr.remote(*args, **kwargs))
        return remote_method