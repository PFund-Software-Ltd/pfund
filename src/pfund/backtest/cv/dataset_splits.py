from __future__ import annotations

from typing import TypedDict

import math

from pfund.backtest.cv.cross_validation import CrossValidator
from pfund.backtest.cv.holdout import Holdout


class DatasetSplitsDict(TypedDict, total=False):
    train: float
    val: float
    dev: float
    test: float


type DatasetSplits = int | DatasetSplitsDict | Holdout | CrossValidator
"""User-facing way to say how to split the dataset.

- ``int``: 3 digits read as train/val/test weights, e.g. ``721`` is 70/20/10.
- ``dict``: at least 2 of ``train``/``val``/``test`` (``dev`` aliases ``val``);
  a missing third ratio takes the remainder.
- ``Holdout`` / ``CrossValidator``: a recipe, passed through as is.
"""


def to_recipe(dataset_splits: DatasetSplits) -> Holdout | CrossValidator:
    """Convert ``dataset_splits`` into a splitting recipe, not bound to any data."""
    if isinstance(dataset_splits, (Holdout, CrossValidator)):
        return dataset_splits
    if isinstance(dataset_splits, bool):
        raise TypeError("`dataset_splits` must not be a bool")
    if isinstance(dataset_splits, int):
        return _holdout_from_int(dataset_splits)
    if isinstance(dataset_splits, dict):
        return _holdout_from_dict(dataset_splits)
    raise TypeError(
        "`dataset_splits` must be an int, a dict, Holdout, or an "
        + f"sklearn-compatible cross-validator, got {type(dataset_splits)}"
    )


def _holdout_from_int(dataset_splits: int) -> Holdout:
    if not 100 <= dataset_splits <= 999:
        raise ValueError(
            "`dataset_splits` int must be 3 digits, e.g. "
            + f'"721" means 70% train, 20% val, 10% test, but got {dataset_splits}'
        )
    digits = [int(digit) for digit in str(dataset_splits)]
    total = sum(digits)
    return Holdout(
        train=digits[0] / total,
        val=digits[1] / total,
        test=digits[2] / total,
    )


def _holdout_from_dict(splits: DatasetSplitsDict) -> Holdout:
    """Build a Holdout from a dict; ratio values are validated by ``Holdout``."""
    unknown_keys = set(splits) - {"train", "val", "dev", "test"}
    if unknown_keys:
        raise ValueError(
            "`dataset_splits` dict contains unsupported key(s): "
            + f"{sorted(unknown_keys)}"
        )
    if "val" in splits and "dev" in splits:
        raise ValueError(
            "`dataset_splits` dict cannot provide both 'val' and its alias 'dev'"
        )

    all_ratios = {
        "train": splits.get("train"),
        "val": splits.get("val", splits.get("dev")),
        "test": splits.get("test"),
    }
    provided = {name: ratio for name, ratio in all_ratios.items() if ratio is not None}
    if len(provided) < 2:
        raise ValueError(
            "`dataset_splits` dict must provide at least 2 of "
            + f"'train'/'val'/'test', but got {sorted(provided)}"
        )

    if len(provided) == 2:
        missing = next(name for name in all_ratios if name not in provided)
        try:
            remainder = 1.0 - sum(provided.values())
        except TypeError:
            remainder = 0.0  # a non-number ratio, which Holdout reports below
        if remainder < 0.0 and not math.isclose(remainder, 0.0, abs_tol=1e-12):
            raise ValueError(
                "`dataset_splits` ratios sum to more than 1.0, cannot derive "
                + f"'{missing}' from {provided}"
            )
        provided[missing] = max(0.0, remainder)
    return Holdout(**provided)
