# ruff: noqa: N802
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

import dataclasses
import math

import numpy as np
import polars as pl
import pytest

from pfund.backtest.cv.holdout import Holdout, resolve_holdout


class TestDefaults:
    def test_default_ratios(self):
        holdout = Holdout()
        assert (holdout.train, holdout.val, holdout.test) == (0.7, 0.2, 0.1)


class TestValid:
    @pytest.mark.parametrize(
        ("train", "val", "test"),
        [
            (1.0, 0.0, 0.0),
            (0.8, 0.2, 0.0),
            (0.8, 0.0, 0.2),
            (0.5, 0.25, 0.25),
            (1, 0, 0),
        ],
        ids=["train_only", "no_test", "no_val", "quarters", "ints"],
    )
    def test_accepts_ratios_summing_to_one(self, train: float, val: float, test: float):
        holdout = Holdout(train=train, val=val, test=test)
        assert (holdout.train, holdout.val, holdout.test) == (train, val, test)

    def test_float_rounding_in_sum_is_tolerated(self):
        # summed in train, val, test order, the defaults miss 1.0 by one ulp
        assert 0.7 + 0.2 + 0.1 != 1.0
        Holdout(train=0.7, val=0.2, test=0.1)


class TestInvalid:
    @pytest.mark.parametrize("field", ["train", "val", "test"])
    @pytest.mark.parametrize("bad", ["0.5", None, [0.5]], ids=["str", "none", "list"])
    def test_non_number_raises(self, field: str, bad: Any):
        with pytest.raises(TypeError, match=f"{field} ratio must be a number"):
            Holdout(**{field: bad})

    @pytest.mark.parametrize("field", ["train", "val", "test"])
    def test_bool_raises(self, field: str):
        with pytest.raises(TypeError, match=f"{field} ratio must be a number"):
            Holdout(**{field: True})

    @pytest.mark.parametrize("field", ["train", "val", "test"])
    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_non_finite_raises(self, field: str, bad: float):
        with pytest.raises(ValueError, match=f"{field} ratio must be finite"):
            Holdout(**{field: bad})

    @pytest.mark.parametrize(
        ("train", "val", "test", "field"),
        [(1.2, -0.2, 0.0, "val"), (1.1, 0.0, -0.1, "test"), (-0.5, 1.0, 0.5, "train")],
    )
    def test_negative_raises(self, train: float, val: float, test: float, field: str):
        with pytest.raises(ValueError, match=f"{field} ratio must be non-negative"):
            Holdout(train=train, val=val, test=test)

    def test_zero_train_raises(self):
        with pytest.raises(ValueError, match="train ratio must be greater than 0"):
            Holdout(train=0.0, val=0.5, test=0.5)

    @pytest.mark.parametrize(
        ("train", "val", "test"),
        [(0.7, 0.2, 0.2), (0.5, 0.2, 0.1), (0.7, 0.2, 0.1 + 1e-6)],
        ids=["over", "under", "slightly_over"],
    )
    def test_not_summing_to_one_raises(self, train: float, val: float, test: float):
        with pytest.raises(ValueError, match=r"holdout ratios must sum to 1\.0"):
            Holdout(train=train, val=val, test=test)


class TestImmutability:
    def test_frozen(self):
        holdout = Holdout()
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(holdout, "train", 0.5)  # noqa: B010

    def test_equality_and_hash(self):
        assert Holdout() == Holdout(train=0.7, val=0.2, test=0.1)
        assert hash(Holdout()) == hash(Holdout(train=0.7, val=0.2, test=0.1))
        assert Holdout() != Holdout(train=0.8, val=0.1, test=0.1)


def test_dev_aliases_val():
    assert Holdout(train=0.6, val=0.3, test=0.1).dev == 0.3


def _make_X(num_dates: int, rows_per_date: int = 1) -> pl.DataFrame:
    dates = [d for d in range(num_dates) for _ in range(rows_per_date)]
    return pl.DataFrame({"date": dates, "row": range(len(dates))})


def _rows(df: Any) -> list[int]:
    return df["row"].to_list()


def _dates(df: Any) -> list[int]:
    return sorted(set(df["date"].to_list()))


class TestResolveHoldout:
    def test_default_ratios_over_ten_dates(self):
        split = resolve_holdout(_make_X(10), Holdout())
        assert _dates(split.X_train) == list(range(7))
        assert split.X_val is not None and _dates(split.X_val) == [7, 8]
        assert split.X_test is not None and _dates(split.X_test) == [9]

    def test_sets_are_chronological(self):
        split = resolve_holdout(_make_X(10), Holdout())
        assert split.X_val is not None and split.X_test is not None
        assert max(_dates(split.X_train)) < min(_dates(split.X_val))
        assert max(_dates(split.X_val)) < min(_dates(split.X_test))

    def test_splits_by_timestamp_not_row(self):
        """All rows sharing a date land in the same set."""
        split = resolve_holdout(_make_X(10, rows_per_date=3), Holdout())
        assert split.X_val is not None and split.X_test is not None
        assert len(split.X_train) == 21
        assert len(split.X_val) == 6
        assert len(split.X_test) == 3
        assert set(_dates(split.X_train)).isdisjoint(_dates(split.X_val))

    def test_partitions_every_row_once(self):
        X = _make_X(7, rows_per_date=2)
        split = resolve_holdout(X, Holdout(train=0.5, val=0.3, test=0.2))
        rows = [
            row
            for df in (split.X_train, split.X_val, split.X_test)
            if df is not None
            for row in _rows(df)
        ]
        assert sorted(rows) == list(range(len(X)))

    def test_rounds_half_up(self):
        # 0.5 * 5 = 2.5 -> 3 train dates, the remaining 2 go to val
        split = resolve_holdout(_make_X(5), Holdout(train=0.5, val=0.5, test=0.0))
        assert _dates(split.X_train) == [0, 1, 2]
        assert split.X_val is not None and _dates(split.X_val) == [3, 4]
        assert split.X_test is None

    def test_zero_ratios_give_none(self):
        split = resolve_holdout(_make_X(4), Holdout(train=1.0, val=0.0, test=0.0))
        assert len(split.X_train) == 4
        assert split.X_val is None
        assert split.X_test is None

    def test_split_applies_to_aligned_labels(self):
        X = _make_X(10)
        y = np.arange(10) * 10
        y_train, y_val, y_test = resolve_holdout(X, Holdout()).split(y)
        assert y_train.tolist() == [0, 10, 20, 30, 40, 50, 60]
        assert y_val is not None and y_val.tolist() == [70, 80]
        assert y_test is not None and y_test.tolist() == [90]

    def test_empty_train_raises(self):
        with pytest.raises(ValueError, match="resulting training set is empty"):
            resolve_holdout(_make_X(10), Holdout(train=0.01, val=0.99, test=0.0))

    def test_empty_val_and_test_warn(self, mocker: MockerFixture):
        cprint = mocker.patch("pfund.backtest.cv.holdout.cprint")
        # 0.96 * 10 rounds to all 10 dates, leaving nothing for val and test
        split = resolve_holdout(_make_X(10), Holdout(train=0.96, val=0.02, test=0.02))
        assert split.X_val is None
        assert split.X_test is None
        messages = [call.args[0] for call in cprint.call_args_list]
        assert len(messages) == 2
        assert messages[0].startswith("validation set is EMPTY")
        assert messages[1].startswith("test set is EMPTY")

    def test_no_warning_when_ratio_is_zero(self, mocker: MockerFixture):
        cprint = mocker.patch("pfund.backtest.cv.holdout.cprint")
        resolve_holdout(_make_X(10), Holdout(train=1.0, val=0.0, test=0.0))
        cprint.assert_not_called()

    def test_requires_date_column(self):
        with pytest.raises(ValueError, match="holdout splitting requires a 'date'"):
            resolve_holdout(pl.DataFrame({"x": [1, 2]}), Holdout())
