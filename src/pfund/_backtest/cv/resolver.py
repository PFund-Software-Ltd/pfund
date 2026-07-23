from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from narwhals.typing import IntoDataFrame
    from numpy.typing import NDArray

import narwhals as nw
import numpy as np
from pfund_kit.style import TextStyle, cprint

from pfund._backtest.cv.base import CrossValidator
from pfund._backtest.cv.dataset_split import DatasetSplit
from pfund._backtest.cv.fold import Fold
from pfund._backtest.cv.holdout import Holdout


def _resolve_time_spine(
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


def _round_half_up(value: float) -> int:
    return math.floor(value + 0.5)


def resolve_holdout(
    X: IntoDataFrame,
    holdout: Holdout,
) -> DatasetSplit:
    """Resolve holdout ratios against the actual unique timestamps in ``X``."""
    time_spine, row_time_indices = _resolve_time_spine(
        X,
        operation="holdout splitting",
    )
    num_times = len(time_spine)

    train_size = _round_half_up(holdout.train * num_times)
    val_size = _round_half_up((holdout.train + holdout.val) * num_times) - train_size
    test_size = num_times - train_size - val_size

    if train_size <= 0:
        raise ValueError(
            "resulting training set is empty "
            + f"(train ratio {holdout.train:.3f} over {num_times} timestamps); "
            + "use a larger `data_range` or a higher train ratio"
        )
    if holdout.val > 0 and val_size <= 0:
        cprint(
            "validation set is EMPTY "
            + f"(val ratio {holdout.val:.3f} over {num_times} timestamps "
            + "rounded to 0); use a larger `data_range` or a higher val ratio",
            style=TextStyle.BOLD,
        )
    if holdout.test > 0 and test_size <= 0:
        cprint(
            "test set is EMPTY "
            + f"(test ratio {holdout.test:.3f} over {num_times} timestamps "
            + "rounded to 0); use a larger `data_range` or a higher test ratio",
            style=TextStyle.BOLD,
        )

    train_end = train_size
    val_end = train_end + val_size

    def take_time_range(start: int, end: int) -> NDArray[np.int64] | None:
        if start == end:
            return None
        return np.flatnonzero(
            (row_time_indices >= start) & (row_time_indices < end)
        ).astype(np.int64, copy=False)

    train_indices = take_time_range(0, train_end)
    assert train_indices is not None
    return DatasetSplit(
        X,
        indices=(
            train_indices,
            take_time_range(train_end, val_end),
            take_time_range(val_end, num_times),
        ),
    )


def resolve_dataset_splits(
    X: IntoDataFrame,
    dataset_splits: Holdout | CrossValidator | None,
) -> DatasetSplit | tuple[Fold, ...]:
    """Bind normalized split configuration to a materialized feature dataset."""
    if dataset_splits is None:
        return DatasetSplit(X)
    if isinstance(dataset_splits, Holdout):
        return resolve_holdout(X, dataset_splits)
    if isinstance(dataset_splits, CrossValidator):
        return resolve_folds(X, dataset_splits)
    raise TypeError(f"unexpected normalized dataset split: {type(dataset_splits)}")


def _expand_time_indices(
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


def resolve_folds(
    X: IntoDataFrame,
    cross_validator: CrossValidator,
) -> tuple[Fold, ...]:
    """Resolve a cross-validator against the actual ordered timestamps in ``X``.

    The cross-validator sees one sample per unique ``date`` value. Its time-level
    selectors are then expanded into positional row selectors, keeping every row
    at the same timestamp in the same side of a fold.
    """
    time_spine, row_time_indices = _resolve_time_spine(
        X,
        operation="cross-validation",
    )
    num_times = len(time_spine)

    folds: list[Fold] = []
    for index, (train_time_indices, val_time_indices) in enumerate(
        cross_validator.split(time_spine)
    ):
        train_indices = _expand_time_indices(
            train_time_indices,
            row_time_indices=row_time_indices,
            num_times=num_times,
            name="train indices",
        )
        val_indices = _expand_time_indices(
            val_time_indices,
            row_time_indices=row_time_indices,
            num_times=num_times,
            name="validation indices",
        )
        folds.append(
            Fold(
                index=index,
                X=X,
                train_indices=train_indices,
                val_indices=val_indices,
            )
        )

    if not folds:
        raise ValueError("cross-validator produced no folds")
    return tuple(folds)
