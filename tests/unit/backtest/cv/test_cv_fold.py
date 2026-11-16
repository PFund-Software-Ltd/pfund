# ruff: noqa: N802, N803
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest

from pfund.backtest.cv.cv_fold import CVFold

N_ROWS = 6


def _idx(*positions: int) -> np.ndarray[Any, np.dtype[np.int64]]:
    return np.array(positions, dtype=np.int64)


def _a(df: Any) -> list[int]:
    """Column ``a`` of a split dataframe as a list."""
    return df["a"].to_list()


@pytest.fixture
def X() -> pl.DataFrame:
    return pl.DataFrame({"a": range(N_ROWS)})


def _fold(X: Any, train: Any = None, val: Any = None, index: int = 0) -> CVFold:
    return CVFold(
        index=index,
        X=X,
        train_indices=_idx(0, 1, 2) if train is None else train,
        val_indices=_idx(3, 4) if val is None else val,
    )


class TestProperties:
    def test_index(self, X: pl.DataFrame):
        assert _fold(X, index=3).index == 3

    def test_indices_are_int64_and_read_only(self, X: pl.DataFrame):
        fold = _fold(X, train=np.array([0, 1], dtype=np.int32))
        assert fold.train_indices.dtype == np.int64
        assert fold.val_indices.dtype == np.int64
        assert not fold.train_indices.flags.writeable
        assert not fold.val_indices.flags.writeable

    def test_X_train_and_X_val(self, X: pl.DataFrame):
        fold = _fold(X)
        assert _a(fold.X_train) == [0, 1, 2]
        assert _a(fold.X_val) == [3, 4]

    def test_may_leave_rows_out(self, X: pl.DataFrame):
        """Gaps/purging: a fold need not cover every row."""
        fold = _fold(X, train=_idx(0, 1), val=_idx(4))
        assert len(fold.X_train) + len(fold.X_val) < N_ROWS


class TestSplit:
    def test_split_numpy(self, X: pl.DataFrame):
        y_train, y_val = _fold(X).split(np.arange(N_ROWS) * 10)
        assert y_train.tolist() == [0, 10, 20]
        assert y_val.tolist() == [30, 40]

    def test_split_keeps_type(self, X: pl.DataFrame):
        y_train, y_val = _fold(X).split(pd.Series(range(N_ROWS)))
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_val, pd.Series)

    def test_split_misaligned_raises(self, X: pl.DataFrame):
        with pytest.raises(ValueError, match="row-aligned"):
            _fold(X).split(np.arange(N_ROWS + 1))


class TestInvalid:
    def test_negative_index_raises(self, X: pl.DataFrame):
        with pytest.raises(ValueError, match="fold index must be non-negative"):
            _fold(X, index=-1)

    def test_overlap_raises(self, X: pl.DataFrame):
        with pytest.raises(ValueError, match="found 2 overlapping row"):
            _fold(X, train=_idx(0, 1, 2), val=_idx(1, 2, 3))

    def test_empty_train_raises(self, X: pl.DataFrame):
        with pytest.raises(ValueError, match="train_indices must not be empty"):
            _fold(X, train=_idx())

    def test_empty_val_raises(self, X: pl.DataFrame):
        with pytest.raises(ValueError, match="val_indices must not be empty"):
            _fold(X, val=_idx())

    def test_out_of_bounds_raises(self, X: pl.DataFrame):
        with pytest.raises(IndexError, match="outside a dataset of 6 rows"):
            _fold(X, val=_idx(6))
