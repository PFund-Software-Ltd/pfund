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
