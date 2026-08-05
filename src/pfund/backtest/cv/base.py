"""The structural interface accepted by pfund's cross-validation resolver."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class CrossValidator(Protocol):
    """Minimal sklearn-compatible splitter interface."""

    def get_n_splits(
        self,
        X: object | None = None,
        y: object | None = None,
        groups: object | None = None,
    ) -> int: ...

    # index arrays are typed Any: sklearn yields numpy ndarrays, and this is an
    # external contract we adapt rather than own.
    def split(
        self, X: object, y: object | None = None, groups: object | None = None
    ) -> Iterator[tuple[Any, Any]]: ...
