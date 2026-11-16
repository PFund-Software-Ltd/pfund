# ruff: noqa: N802, N803
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pytest_mock import MockerFixture

import dataclasses

import numpy as np
import polars as pl
import pytest
from sklearn.model_selection import KFold, TimeSeriesSplit

from pfund.backtest.cv.cross_validation import (
    CrossValidation,
    CrossValidator,
    resolve_folds,
)


def _make_X(num_dates: int, rows_per_date: int = 1) -> pl.DataFrame:
    dates = [d for d in range(num_dates) for _ in range(rows_per_date)]
    return pl.DataFrame({"date": dates, "row": range(len(dates))})


def _dates(df: Any) -> list[int]:
    return sorted(set(df["date"].to_list()))


class TestCrossValidationParams:
    def test_defaults(self):
        cv = CrossValidation()
        assert (cv.n_splits, cv.max_train_size, cv.test_size, cv.gap) == (
            5,
            None,
            None,
            0,
        )

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"n_splits": 1}, "n_splits must be at least 2"),
            ({"max_train_size": 0}, "max_train_size must be at least 1"),
            ({"test_size": 0}, "test_size must be at least 1"),
            ({"gap": -1}, "gap must be at least 0"),
        ],
    )
    def test_out_of_range_raises(self, kwargs: dict[str, int], match: str):
        with pytest.raises(ValueError, match=match):
            CrossValidation(**kwargs)

    @pytest.mark.parametrize(
        "field", ["n_splits", "max_train_size", "test_size", "gap"]
    )
    @pytest.mark.parametrize("bad", [True, 2.0, "2"], ids=["bool", "float", "str"])
    def test_non_int_raises(self, field: str, bad: Any):
        with pytest.raises(TypeError, match=f"{field} must be an int"):
            CrossValidation(**{field: bad})

    def test_frozen(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(CrossValidation(), "n_splits", 3)  # noqa: B010


class TestCrossValidationSplitter:
    def test_is_a_cross_validator(self):
        assert isinstance(CrossValidation(), CrossValidator)

    def test_get_n_splits(self):
        assert CrossValidation(n_splits=4).get_n_splits() == 4

    @pytest.mark.parametrize(
        "kwargs",
        [{}, {"n_splits": 3}, {"max_train_size": 3}, {"test_size": 2}, {"gap": 1}],
    )
    def test_matches_sklearn_time_series_split(self, kwargs: dict[str, int]):
        samples = np.arange(20)
        ours = list(CrossValidation(**kwargs).split(samples))
        theirs = list(TimeSeriesSplit(**kwargs).split(samples))
        assert len(ours) == len(theirs)
        for (a_train, a_val), (b_train, b_val) in zip(ours, theirs, strict=True):
            assert a_train.tolist() == b_train.tolist()
            assert a_val.tolist() == b_val.tolist()

    def test_missing_sklearn_raises(self, mocker: MockerFixture):
        mocker.patch.dict("sys.modules", {"sklearn.model_selection": None})
        with pytest.raises(ImportError, match="install pfund with the 'sklearn' extra"):
            list(CrossValidation().split(np.arange(10)))


class TestCrossValidatorProtocol:
    def test_sklearn_splitters_conform(self):
        assert isinstance(KFold(), CrossValidator)
        assert isinstance(TimeSeriesSplit(), CrossValidator)

    def test_duck_typed_splitter_conforms(self):
        class Splitter:
            def get_n_splits(self, X=None, y=None, groups=None) -> int:
                return 1

            def split(self, X, y=None, groups=None) -> Iterator[tuple[Any, Any]]:
                yield np.array([0]), np.array([1])

        assert isinstance(Splitter(), CrossValidator)

    def test_object_without_split_does_not_conform(self):
        assert not isinstance(object(), CrossValidator)


class TestResolveFolds:
    def test_matches_time_series_split_on_dates(self):
        X = _make_X(10)
        folds = resolve_folds(X, CrossValidation(n_splits=3))
        expected = list(TimeSeriesSplit(n_splits=3).split(np.arange(10)))
        assert [fold.index for fold in folds] == [0, 1, 2]
        for fold, (train, val) in zip(folds, expected, strict=True):
            assert _dates(fold.X_train) == train.tolist()
            assert _dates(fold.X_val) == val.tolist()

    def test_splits_by_timestamp_not_row(self):
        """The splitter sees one sample per date; every row of a date follows it."""
        X = _make_X(10, rows_per_date=3)
        folds = resolve_folds(X, CrossValidation(n_splits=3))
        assert len(folds) == 3
        for fold in folds:
            assert len(fold.X_train) == 3 * len(_dates(fold.X_train))
            assert len(fold.X_val) == 3 * len(_dates(fold.X_val))
            assert set(_dates(fold.X_train)).isdisjoint(_dates(fold.X_val))

    def test_gap_leaves_rows_out(self):
        X = _make_X(10, rows_per_date=2)
        for fold in resolve_folds(X, CrossValidation(n_splits=3, gap=1)):
            assert max(_dates(fold.X_train)) + 1 < min(_dates(fold.X_val))

    def test_accepts_any_sklearn_splitter(self):
        """Non-chronological splitters (e.g. KFold) are passed through as given."""
        X = _make_X(6, rows_per_date=2)
        folds = resolve_folds(X, KFold(n_splits=3))
        assert [_dates(fold.X_val) for fold in folds] == [[0, 1], [2, 3], [4, 5]]
        assert _dates(folds[1].X_train) == [0, 1, 4, 5]

    def test_split_applies_to_aligned_labels(self):
        X = _make_X(6)
        y = np.arange(6) * 10
        fold = resolve_folds(X, CrossValidation(n_splits=2))[0]
        y_train, y_val = fold.split(y)
        assert y_train.tolist() == [0, 10]
        assert y_val.tolist() == [20, 30]

    def test_no_folds_raises(self):
        class EmptySplitter:
            def get_n_splits(self, X=None, y=None, groups=None) -> int:
                return 0

            def split(self, X, y=None, groups=None) -> Iterator[tuple[Any, Any]]:
                yield from ()

        with pytest.raises(ValueError, match="cross-validator produced no folds"):
            resolve_folds(_make_X(5), EmptySplitter())

    def test_bad_splitter_output_raises(self):
        class OutOfRangeSplitter:
            def get_n_splits(self, X=None, y=None, groups=None) -> int:
                return 1

            def split(self, X, y=None, groups=None) -> Iterator[tuple[Any, Any]]:
                yield np.array([0]), np.array([len(X)])

        with pytest.raises(IndexError, match="validation indices contains a position"):
            resolve_folds(_make_X(5), OutOfRangeSplitter())

    def test_requires_date_column(self):
        with pytest.raises(ValueError, match="cross-validation requires a 'date'"):
            resolve_folds(pl.DataFrame({"x": [1, 2]}), CrossValidation(n_splits=2))
