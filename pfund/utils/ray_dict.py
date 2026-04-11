# VIBE-CODED
from __future__ import annotations

from collections.abc import Iterator
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ray.actor import ActorHandle


class _SharedDictActor:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def getitem(self, key: str) -> tuple[bool, Any]:
        if key in self._data:
            return True, self._data[key]
        return False, None

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    def delete(self, key: str) -> None:
        del self._data[key]

    def contains(self, key: str) -> bool:
        return key in self._data

    def get_all(self) -> dict[str, Any]:
        return self._data

    def update(self, d: dict[str, Any]) -> None:
        self._data.update(d)

    def keys(self) -> list[str]:
        return list(self._data.keys())

    def values(self) -> list[Any]:
        return list(self._data.values())

    def items(self) -> list[tuple[str, Any]]:
        return list(self._data.items())

    def len(self) -> int:
        return len(self._data)


_RemoteSharedDictActor: type | None = None


def _get_remote_actor_class() -> type:
    global _RemoteSharedDictActor
    if _RemoteSharedDictActor is None:
        import ray
        _RemoteSharedDictActor = ray.remote(_SharedDictActor)
    assert _RemoteSharedDictActor is not None
    return _RemoteSharedDictActor


class RayCompatibleDict:
    """A dict-like class that works as a normal dict by default,
    but can switch to a Ray actor-backed mode for cross-process shared state.
    """

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        self._data: dict[str, Any] | None = dict(data) if data else {}
        self._actor: ActorHandle[_SharedDictActor] | None = None

    @property
    def _local_data(self) -> dict[str, Any]:
        """Access _data with a None guard. Only valid when ray is disabled."""
        assert self._data is not None, "Cannot access local data when ray is enabled"
        return self._data

    def enable_ray(self) -> None:
        """Switch to Ray-backed mode. Syncs current data to the actor."""
        if self._actor is not None:
            return
        actor_cls = _get_remote_actor_class()
        self._actor = actor_cls.remote(self._data)
        self._data = None

    def disable_ray(self) -> None:
        """Switch back to local mode. Pulls data from the actor."""
        if self._actor is not None:
            import ray
            self._data = ray.get(self._actor.get_all.remote())
            self._actor = None

    @property
    def is_ray_enabled(self) -> bool:
        return self._actor is not None

    def _ray_get(self, ref: Any) -> Any:
        import ray
        return ray.get(ref)

    def __getitem__(self, key: str) -> Any:
        if self._actor is not None:
            found, val = self._ray_get(self._actor.getitem.remote(key))
            if not found:
                raise KeyError(key)
            return val
        return self._local_data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        if self._actor is not None:
            self._ray_get(self._actor.set.remote(key, value))
        else:
            self._local_data[key] = value

    def __delitem__(self, key: str) -> None:
        if self._actor is not None:
            self._ray_get(self._actor.delete.remote(key))
        else:
            del self._local_data[key]

    def __contains__(self, key: object) -> bool:
        if self._actor is not None:
            return self._ray_get(self._actor.contains.remote(key))
        return key in self._local_data

    def __len__(self) -> int:
        if self._actor is not None:
            return self._ray_get(self._actor.len.remote())
        return len(self._local_data)

    def __iter__(self) -> Iterator[str]:
        if self._actor is not None:
            return iter(self._ray_get(self._actor.keys.remote()))
        return iter(self._local_data)

    def __repr__(self) -> str:
        data = self._ray_get(self._actor.get_all.remote()) if self._actor is not None else self._local_data
        return f"RayCompatibleDict({data})"

    def __bool__(self) -> bool:
        return len(self) > 0

    def __eq__(self, other: object) -> bool:
        if isinstance(other, RayCompatibleDict):
            return self.to_dict() == other.to_dict()
        if isinstance(other, dict):
            return self.to_dict() == other
        return NotImplemented

    def get(self, key: str, default: Any = None) -> Any:
        if self._actor is not None:
            return self._ray_get(self._actor.get.remote(key, default))
        return self._local_data.get(key, default)

    def update(self, d: dict[str, Any]) -> None:
        if self._actor is not None:
            self._ray_get(self._actor.update.remote(d))
        else:
            self._local_data.update(d)

    def keys(self) -> list[str]:
        if self._actor is not None:
            return self._ray_get(self._actor.keys.remote())
        return list(self._local_data.keys())

    def values(self) -> list[Any]:
        if self._actor is not None:
            return self._ray_get(self._actor.values.remote())
        return list(self._local_data.values())

    def items(self) -> list[tuple[str, Any]]:
        if self._actor is not None:
            return self._ray_get(self._actor.items.remote())
        return list(self._local_data.items())

    def to_dict(self) -> dict[str, Any]:
        if self._actor is not None:
            return self._ray_get(self._actor.get_all.remote())
        return self._local_data.copy()
