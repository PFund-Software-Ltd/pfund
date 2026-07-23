from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from narwhals.typing import IntoDataFrame
    from numpy.typing import NDArray

import numpy as np

from pfund._backtest.cv.indexing import (
    DataT,
    normalize_indices,
    row_count,
    take_rows,
    validate_aligned,
)


class Fold:
    """One resolved cross-validation train/validation split.

    A fold is bound to a materialized feature dataframe and stores only
    positional row selectors. The dataframe slices are produced lazily so
    constructing folds does not duplicate the complete feature dataset.
    """

    def __init__(
        self,
        *,
        index: int,
        X: IntoDataFrame,
        train_indices: NDArray[Any],
        val_indices: NDArray[Any],
    ):
        if index < 0:
            raise ValueError(f"fold index must be non-negative, got {index}")

        n_rows = row_count(X)
        train_indices = normalize_indices(
            train_indices,
            name="train_indices",
            n_rows=n_rows,
        )
        val_indices = normalize_indices(
            val_indices,
            name="val_indices",
            n_rows=n_rows,
        )
        overlap = np.intersect1d(
            train_indices,
            val_indices,
            assume_unique=True,
        )
        if overlap.size:
            raise ValueError(
                "train_indices and val_indices must not overlap; "
                + f"found {overlap.size} overlapping row(s)"
            )

        self._index = index
        self._X = X
        self._train_indices = train_indices
        self._val_indices = val_indices

    @property
    def index(self) -> int:
        return self._index

    @property
    def train_indices(self) -> NDArray[np.int64]:
        return self._train_indices

    @property
    def val_indices(self) -> NDArray[np.int64]:
        return self._val_indices

    @property
    def X_train(self) -> IntoDataFrame:
        return take_rows(self._X, self._train_indices)

    @property
    def X_val(self) -> IntoDataFrame:
        return take_rows(self._X, self._val_indices)

    def split(self, data: DataT) -> tuple[DataT, DataT]:
        """Apply this fold's selectors to data aligned row-for-row with ``X``."""
        validate_aligned(data, self._X)
        return (
            take_rows(data, self._train_indices),
            take_rows(data, self._val_indices),
        )
