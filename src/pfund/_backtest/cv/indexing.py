from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar, cast

if TYPE_CHECKING:
    from narwhals.typing import IntoDataFrame, IntoSeries
    from numpy.typing import NDArray

    DataT = TypeVar("DataT", IntoDataFrame, IntoSeries, "NDArray[Any]")
else:
    DataT = TypeVar("DataT")

import narwhals as nw
import numpy as np


def row_count(data: DataT) -> int:
    if isinstance(data, np.ndarray):
        if data.ndim == 0:
            raise ValueError("`data` must have at least one dimension")
        return data.shape[0]
    return len(nw.from_native(data, allow_series=True))


def normalize_indices(
    indices: NDArray[Any],
    *,
    name: str,
    n_rows: int,
) -> NDArray[np.int64]:
    """Own and validate an immutable array of positional row selectors."""
    indices = np.asarray(indices)
    if indices.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if indices.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.issubdtype(indices.dtype, np.integer):
        raise TypeError(f"{name} must contain integer row positions")

    indices = indices.astype(np.int64, copy=True)
    if np.any(indices < 0) or np.any(indices >= n_rows):
        raise IndexError(
            f"{name} contains a row position outside a dataset of {n_rows} rows"
        )
    if np.unique(indices).size != indices.size:
        raise ValueError(f"{name} must not contain duplicate row positions")

    indices.flags.writeable = False
    return indices


def validate_aligned(data: DataT, reference: IntoDataFrame) -> None:
    n_ref = row_count(reference)
    n_data = row_count(data)
    if n_data != n_ref:
        raise ValueError(
            "`data` must be row-aligned 1:1 with the feature dataset "
            + f"({n_ref} rows), but got {n_data} rows"
        )


def take_rows(data: DataT, indices: NDArray[np.int64]) -> DataT:
    if isinstance(data, np.ndarray):
        return cast("DataT", data[indices])
    return cast(
        "DataT",
        nw.from_native(data, allow_series=True)[indices.tolist()].to_native(),
    )
