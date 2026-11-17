from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from narwhals.typing import IntoDataFrame, IntoSeries
    from numpy.typing import NDArray

import narwhals as nw
import numpy as np

# data that can be split row-for-row alongside the feature dataset X
type SplittableData = IntoDataFrame | IntoSeries | NDArray[Any]


def row_count(data: SplittableData) -> int:
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


def validate_aligned(data: SplittableData, reference: IntoDataFrame) -> None:
    n_ref = row_count(reference)
    n_data = row_count(data)
    if n_data != n_ref:
        raise ValueError(
            "`data` must be row-aligned 1:1 with the feature dataset "
            + f"({n_ref} rows), but got {n_data} rows"
        )


def take_rows[DataT: SplittableData](data: DataT, indices: NDArray[np.int64]) -> DataT:
    if isinstance(data, np.ndarray):
        return cast("DataT", data[indices])
    return cast(
        "DataT",
        nw.from_native(data, allow_series=True)[indices.tolist()].to_native(),
    )


def resolve_time_spine(
    X: IntoDataFrame,
    *,
    operation: str,
) -> tuple[NDArray[Any], NDArray[np.int64]]:
    """Return unique ordered timestamps and each row's timestamp position."""
    df = nw.from_native(X)
    if "date" not in df.columns:
        raise ValueError(f"{operation} requires a 'date' column in X")

    date = df.get_column("date")
    if len(date) == 0:
        raise ValueError(f"{operation} requires at least one row in X")
    if date.is_null().any():
        raise ValueError(f"{operation} does not support null dates in X")
    if not date.is_sorted():
        raise ValueError(f"X must be sorted by 'date' before {operation}")

    row_dates = date.to_numpy()
    starts_new_time = np.empty(len(row_dates), dtype=bool)
    starts_new_time[0] = True
    starts_new_time[1:] = row_dates[1:] != row_dates[:-1]

    time_spine = row_dates[starts_new_time]
    row_time_indices = np.cumsum(starts_new_time, dtype=np.int64) - 1
    return time_spine, row_time_indices


def expand_time_indices(
    time_indices: NDArray[Any],
    *,
    row_time_indices: NDArray[np.int64],
    num_times: int,
    name: str,
) -> NDArray[np.int64]:
    time_indices = np.asarray(time_indices)
    if time_indices.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if time_indices.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.issubdtype(time_indices.dtype, np.integer):
        raise TypeError(f"{name} must contain integer positions")
    if np.any(time_indices < 0) or np.any(time_indices >= num_times):
        raise IndexError(
            f"{name} contains a position outside a spine of {num_times} timestamps"
        )
    if np.unique(time_indices).size != time_indices.size:
        raise ValueError(f"{name} must not contain duplicate positions")

    selected_times = np.zeros(num_times, dtype=bool)
    selected_times[time_indices] = True
    return np.flatnonzero(selected_times[row_time_indices])
