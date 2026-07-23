from __future__ import annotations

from typing import TYPE_CHECKING, cast

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


class DatasetSplit:
    """One component-bound train/validation/test partition.

    With no positional selectors, the complete feature dataset is the training
    set. Cross-validation is represented separately by ``Fold`` objects.
    """

    def __init__(
        self,
        X: IntoDataFrame,
        *,
        indices: tuple[
            NDArray[np.int64],
            NDArray[np.int64] | None,
            NDArray[np.int64] | None,
        ]
        | None = None,
    ):
        self._X = X
        if indices is None:
            self._indices = None
            return

        n_rows = row_count(X)
        train_indices, val_indices, test_indices = indices
        normalized_indices = (
            normalize_indices(
                train_indices,
                name="train_indices",
                n_rows=n_rows,
            ),
            None
            if val_indices is None
            else normalize_indices(
                val_indices,
                name="val_indices",
                n_rows=n_rows,
            ),
            None
            if test_indices is None
            else normalize_indices(
                test_indices,
                name="test_indices",
                n_rows=n_rows,
            ),
        )

        assigned_indices = np.concatenate(
            [index for index in normalized_indices if index is not None]
        )
        num_unique_indices = np.unique(assigned_indices).size
        if num_unique_indices != assigned_indices.size:
            raise ValueError("train, validation, and test indices must not overlap")
        if num_unique_indices != n_rows:
            raise ValueError(
                "train, validation, and test indices must partition every row "
                + f"exactly once; assigned {num_unique_indices} of {n_rows} rows"
            )

        self._indices = normalized_indices

    def _get_X_split(self, index: int) -> IntoDataFrame | None:
        if self._indices is None:
            return self._X if index == 0 else None
        indices = self._indices[index]
        return None if indices is None else take_rows(self._X, indices)

    @property
    def X_train(self) -> IntoDataFrame:
        X_train = self._get_X_split(0)
        assert X_train is not None
        return X_train

    @property
    def X_val(self) -> IntoDataFrame | None:
        return self._get_X_split(1)

    @property
    def X_test(self) -> IntoDataFrame | None:
        return self._get_X_split(2)

    def split(
        self,
        data: DataT,
    ) -> tuple[DataT, DataT | None, DataT | None]:
        """Apply the partition to data aligned row-for-row with ``X``."""
        validate_aligned(data, self._X)
        if self._indices is None:
            return data, None, None

        return cast(
            "tuple[DataT, DataT | None, DataT | None]",
            tuple(
                None if indices is None else take_rows(data, indices)
                for indices in self._indices
            ),
        )
