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
