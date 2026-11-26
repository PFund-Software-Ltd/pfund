from __future__ import annotations

from typing import Any

import dataclasses
import math

import pytest

from pfund.backtest.cv.holdout import Holdout


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
