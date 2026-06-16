from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ray.actor import ActorHandle


# Identity of the one shared actor. Name + namespace is how every Ray actor
# and the driver resolve the *same* registry via `ray.get_actor`.
REGISTRY_ACTOR_NAME = "pfund_component_registry"
REGISTRY_NAMESPACE = "pfund"


class ComponentRegistry:
    """Tracks component names that are in use, enforcing global uniqueness.

    This is a plain class on purpose: it knows nothing about Ray. The exact
    same class is wrapped by `ray.remote(...)` when any component runs remotely,
    so its method bodies must be correct both in-process and inside an actor.
    Inside an actor, method calls are serialized (single-threaded), which makes
    `claim`'s check-and-insert atomic across competing callers.
    """

    def __init__(self):
        self._names: set[str] = set()

    def claim(self, name: str) -> None:
        if name in self._names:
            raise ValueError(f"Component name '{name}' is already in use globally")
        self._names.add(name)

    def release(self, name: str) -> None:
        # discard, not remove: releasing an unclaimed name during teardown must not raise
        self._names.discard(name)

    def names(self) -> list[str]:
        # return a copy: the value is pickled across the wire in actor mode,
        # and callers must never mutate internal state by reference
        return list(self._names)


def get_or_create_registry_actor(lifetime: str | None = None) -> ActorHandle:
    """Get-or-create the single shared registry actor.

    Wraps the plain `ComponentRegistry` with `ray.remote` and pins it to a fixed
    name + namespace so every actor and the driver resolve the same instance.

    Args:
        lifetime:
            - None (default): actor dies with the Ray job -> names are scoped
              per-run and never leak between separate `ray.init` sessions.
            - "detached": actor outlives the driver -> claims persist across jobs.
    """
    import ray

    RegistryActor = ray.remote(num_cpus=0)(ComponentRegistry)
    return RegistryActor.options(
        name=REGISTRY_ACTOR_NAME,
        namespace=REGISTRY_NAMESPACE,
        lifetime=lifetime,
        get_if_exists=True,  # race-safe: concurrent callers share one actor instead of erroring
    ).remote()


class RegistryProxy:
    """Local stand-in for the registry actor that mirrors `ComponentRegistry`'s API.

    Hides the `ray.get(handle.method.remote(...))` round-trip so call sites stay
    identical whether the registry is a plain object or a remote actor. Holds only
    the actor handle, which is serializable, so the proxy survives being pickled
    into the engine context that ships to each actor — and every copy resolves the
    same underlying actor.

    A dedicated proxy (rather than reusing `ActorProxy`) keeps this free of the
    component-specific baggage `ActorProxy` requires (resolution, component_type,
    num_cpus, ZMQ wiring).
    """

    def __init__(self, handle: ActorHandle):
        self._actor = handle

    def claim(self, name: str) -> None:
        import ray

        ray.get(self._actor.claim.remote(name))

    def release(self, name: str) -> None:
        import ray

        ray.get(self._actor.release.remote(name))

    def names(self) -> list[str]:
        import ray

        return ray.get(self._actor.names.remote())


def to_registry_proxy(
    registry: ComponentRegistry | RegistryProxy,
    lifetime: str | None = None,
) -> RegistryProxy:
    """Upgrade a local registry to the shared actor-backed one, preserving claims.

    MUST be called before the first remote component's `ActorProxy` is created, so
    the upgraded (proxy-holding) engine context is what gets pickled and shipped to
    the actor. A strategy actor handed a context with a *plain* registry can never
    retroactively share — its later claims would be invisible to other actors.

    Idempotent: a registry that is already distributed is returned unchanged.
    Any names claimed locally before the upgrade are migrated into the actor.
    """
    if isinstance(registry, RegistryProxy):
        return registry

    proxy = RegistryProxy(get_or_create_registry_actor(lifetime=lifetime))
    # replay pre-upgrade local claims; the freshly created actor is empty, so no collisions
    for name in registry.names():
        proxy.claim(name)
    return proxy
