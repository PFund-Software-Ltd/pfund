# ruff: noqa: N802, N803
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest

from pfund.backtest.cv.holdout_split import HoldoutSplit

N_ROWS = 6


def _idx(*positions: int) -> np.ndarray[Any, np.dtype[np.int64]]:
    return np.array(positions, dtype=np.int64)


def _a(df: Any) -> list[int]:
    """Column ``a`` of a split dataframe as a list."""
    return df["a"].to_list()


@pytest.fixture
def X() -> pl.DataFrame:
    return pl.DataFrame({"a": range(N_ROWS)})


class TestNoIndices:
    def test_whole_dataset_is_train(self, X: pl.DataFrame):
        split = HoldoutSplit(X)
        assert split.X_train is X
        assert split.X_val is None
        assert split.X_test is None

    def test_split_returns_data_unchanged(self, X: pl.DataFrame):
        y = np.arange(N_ROWS)
        y_train, y_val, y_test = HoldoutSplit(X).split(y)
        assert y_train is y
        assert y_val is None
        assert y_test is None

    def test_split_still_checks_alignment(self, X: pl.DataFrame):
        with pytest.raises(ValueError, match="row-aligned"):
            HoldoutSplit(X).split(np.arange(N_ROWS + 1))


class TestWithIndices:
    def test_selects_rows(self, X: pl.DataFrame):
        split = HoldoutSplit(X, indices=(_idx(0, 1, 2), _idx(3, 4), _idx(5)))
        assert _a(split.X_train) == [0, 1, 2]
        assert split.X_val is not None and _a(split.X_val) == [3, 4]
        assert split.X_test is not None and _a(split.X_test) == [5]

    @pytest.mark.parametrize(
        ("val", "test"),
        [(None, _idx(4, 5)), (_idx(4, 5), None)],
        ids=["no_val", "no_test"],
    )
    def test_optional_sets(self, X: pl.DataFrame, val: Any, test: Any):
        split = HoldoutSplit(X, indices=(_idx(0, 1, 2, 3), val, test))
        assert (split.X_val is None) == (val is None)
        assert (split.X_test is None) == (test is None)

    def test_train_only(self, X: pl.DataFrame):
        split = HoldoutSplit(X, indices=(_idx(*range(N_ROWS)), None, None))
        assert _a(split.X_train) == list(range(N_ROWS))
        assert split.X_val is None
        assert split.X_test is None

    def test_split_applies_to_each_container(self, X: pl.DataFrame):
        split = HoldoutSplit(X, indices=(_idx(0, 1, 2), _idx(3, 4), _idx(5)))
        y = pd.Series(range(N_ROWS)) * 10
        y_train, y_val, y_test = split.split(y)
        assert isinstance(y_train, pd.Series)
        assert y_train.tolist() == [0, 10, 20]
        assert y_val is not None and y_val.tolist() == [30, 40]
        assert y_test is not None and y_test.tolist() == [50]

    def test_split_keeps_none_for_missing_sets(self, X: pl.DataFrame):
        split = HoldoutSplit(X, indices=(_idx(0, 1, 2, 3), None, _idx(4, 5)))
        _, y_val, y_test = split.split(np.arange(N_ROWS))
        assert y_val is None
        assert y_test is not None and y_test.tolist() == [4, 5]

    def test_split_misaligned_raises(self, X: pl.DataFrame):
        split = HoldoutSplit(X, indices=(_idx(0, 1, 2), _idx(3, 4), _idx(5)))
        with pytest.raises(ValueError, match="row-aligned"):
            split.split(np.arange(N_ROWS - 1))

    def test_caller_mutation_does_not_leak(self, X: pl.DataFrame):
        train = _idx(0, 1, 2)
        split = HoldoutSplit(X, indices=(train, _idx(3, 4), _idx(5)))
        train[0] = 5
        assert _a(split.X_train) == [0, 1, 2]


class TestInvalidIndices:
    def test_overlap_raises(self, X: pl.DataFrame):
        with pytest.raises(ValueError, match="must not overlap"):
            HoldoutSplit(X, indices=(_idx(0, 1, 2, 3), _idx(3, 4), _idx(5)))

    def test_missing_rows_raise(self, X: pl.DataFrame):
        with pytest.raises(ValueError, match="assigned 5 of 6 rows"):
            HoldoutSplit(X, indices=(_idx(0, 1, 2), _idx(3, 4), None))

    @pytest.mark.parametrize(
        ("indices", "name"),
        [
            ((_idx(), _idx(0, 1, 2), _idx(3, 4, 5)), "train_indices"),
            ((_idx(0, 1, 2), _idx(), _idx(3, 4, 5)), "val_indices"),
            ((_idx(0, 1, 2), _idx(3, 4, 5), _idx()), "test_indices"),
        ],
    )
    def test_empty_set_raises_with_its_name(
        self, X: pl.DataFrame, indices: Any, name: str
    ):
        with pytest.raises(ValueError, match=f"{name} must not be empty"):
            HoldoutSplit(X, indices=indices)

    def test_out_of_bounds_raises(self, X: pl.DataFrame):
        with pytest.raises(IndexError, match="outside a dataset of 6 rows"):
            HoldoutSplit(X, indices=(_idx(0, 1, 2, 3, 4, 6), None, None))
