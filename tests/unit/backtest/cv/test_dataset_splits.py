from __future__ import annotations

from typing import Any

import math

import pytest

from pfund.backtest.cv.cross_validation import CrossValidation
from pfund.backtest.cv.dataset_splits import to_recipe
from pfund.backtest.cv.holdout import Holdout


def _ratios(holdout: Any) -> tuple[float, float, float]:
    assert isinstance(holdout, Holdout)
    return holdout.train, holdout.val, holdout.test


class TestPassThrough:
    def test_holdout(self):
        holdout = Holdout(train=0.8, val=0.1, test=0.1)
        assert to_recipe(holdout) is holdout

    def test_cross_validator(self):
        cv = CrossValidation(n_splits=3)
        assert to_recipe(cv) is cv

    def test_sklearn_splitter(self):
        from sklearn.model_selection import KFold

        kfold = KFold(n_splits=3)
        assert to_recipe(kfold) is kfold


class TestInt:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [(721, (0.7, 0.2, 0.1)), (100, (1.0, 0.0, 0.0)), (802, (0.8, 0.0, 0.2))],
    )
    def test_digits_are_tenths(self, value: int, expected: tuple[float, ...]):
        assert _ratios(to_recipe(value)) == pytest.approx(expected)

    def test_digits_are_normalized_by_their_sum(self):
        # 1 + 1 + 2 = 4
        assert _ratios(to_recipe(112)) == pytest.approx((0.25, 0.25, 0.5))

    @pytest.mark.parametrize("value", [99, 1000, -721, 0])
    def test_not_3_digits_raises(self, value: int):
        with pytest.raises(ValueError, match="int must be 3 digits"):
            to_recipe(value)

    def test_bool_raises(self):
        with pytest.raises(TypeError, match="must not be a bool"):
            to_recipe(True)


class TestDict:
    def test_all_three(self):
        recipe = to_recipe({"train": 0.6, "val": 0.3, "test": 0.1})
        assert _ratios(recipe) == pytest.approx((0.6, 0.3, 0.1))

    @pytest.mark.parametrize(
        ("splits", "expected"),
        [
            ({"train": 0.8, "test": 0.2}, (0.8, 0.0, 0.2)),
            ({"train": 0.8, "val": 0.2}, (0.8, 0.2, 0.0)),
            ({"val": 0.2, "test": 0.1}, (0.7, 0.2, 0.1)),
        ],
        ids=["no_val", "no_test", "no_train"],
    )
    def test_missing_ratio_takes_remainder(
        self, splits: Any, expected: tuple[float, ...]
    ):
        assert _ratios(to_recipe(splits)) == pytest.approx(expected)

    def test_remainder_rounding_noise_is_clamped_to_zero(self):
        # 0.1 + 0.2 + 0.7 overshoots 1.0 by one ulp
        recipe = to_recipe({"train": 0.1 + 0.2, "val": 0.7})
        assert _ratios(recipe)[2] == 0.0

    def test_dev_aliases_val(self):
        recipe = to_recipe({"train": 0.7, "dev": 0.2, "test": 0.1})
        assert _ratios(recipe) == pytest.approx((0.7, 0.2, 0.1))

    def test_val_and_dev_raises(self):
        with pytest.raises(ValueError, match="both 'val' and its alias 'dev'"):
            to_recipe({"train": 0.7, "val": 0.2, "dev": 0.2})

    def test_unknown_key_raises(self):
        with pytest.raises(ValueError, match=r"unsupported key\(s\): \['valid'\]"):
            splits: Any = {"train": 0.8, "valid": 0.2}
            to_recipe(splits)

    @pytest.mark.parametrize("splits", [{}, {"train": 1.0}])
    def test_fewer_than_two_raises(self, splits: Any):
        with pytest.raises(ValueError, match="must provide at least 2"):
            to_recipe(splits)

    def test_two_summing_over_one_raises(self):
        with pytest.raises(ValueError, match="cannot derive 'test'"):
            to_recipe({"train": 0.8, "val": 0.3})

    def test_three_not_summing_to_one_raises(self):
        with pytest.raises(ValueError, match=r"holdout ratios must sum to 1\.0"):
            to_recipe({"train": 0.7, "val": 0.2, "test": 0.2})

    # ratio values are validated by Holdout
    @pytest.mark.parametrize("bad", ["0.2", True], ids=["str", "bool"])
    def test_non_number_raises(self, bad: Any):
        with pytest.raises(TypeError, match="val ratio must be a number"):
            to_recipe({"train": 0.8, "val": bad, "test": 0.2})

    def test_nan_raises(self):
        with pytest.raises(ValueError, match="val ratio must be finite"):
            to_recipe({"train": 0.8, "val": math.nan})

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="val ratio must be non-negative"):
            to_recipe({"train": 0.8, "val": -0.1})

    def test_zero_train_raises(self):
        with pytest.raises(ValueError, match="train ratio must be greater than 0"):
            to_recipe({"val": 0.5, "test": 0.5})


@pytest.mark.parametrize(
    "bad", ["721", 0.7, [0.7, 0.2, 0.1]], ids=["str", "float", "list"]
)
def test_unsupported_type_raises(bad: Any):
    with pytest.raises(TypeError, match="must be an int, a dict"):
        to_recipe(bad)
