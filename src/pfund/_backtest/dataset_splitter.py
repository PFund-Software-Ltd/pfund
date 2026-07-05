from __future__ import annotations

from typing import TypedDict, TYPE_CHECKING

if TYPE_CHECKING:
    from sklearn.model_selection import TimeSeriesSplit

    # canonical home is cv.base; re-exported from this module for back-compat
    from pfund._backtest.cv.base import CrossValidator, CrossValidatorDatasetPeriods

    class DatasetSplitsDict(TypedDict, total=False):
        train: float
        dev: float
        test: float


import math
import datetime
from dataclasses import dataclass, field

try:
    # from sklearn.model_selection._split import BaseCrossValidator
    from sklearn.model_selection import TimeSeriesSplit as _TimeSeriesSplit
except ImportError:
    _TimeSeriesSplit = None


class DatasetPeriods(TypedDict):
    dataset: tuple[datetime.date, datetime.date]
    train_set: tuple[datetime.date, datetime.date]
    dev_set: tuple[datetime.date | None, datetime.date | None]
    test_set: tuple[datetime.date | None, datetime.date | None]


@dataclass(frozen=True)
class DatasetSplitter:
    """
    A dataclass that splits a dataset (based on `data_range`) into train/dev/test,
    either by ratio or by TimeSeriesSplit.
    """

    dataset_start: datetime.date
    dataset_end: datetime.date
    dataset_splits: int | DatasetSplitsDict | TimeSeriesSplit = 721

    # Derived fields:
    dataset_periods: DatasetPeriods | list[CrossValidatorDatasetPeriods] = field(
        init=False
    )

    def __post_init__(self):
        if isinstance(self.dataset_splits, (int, dict)):
            dataset_periods = self._split_by_ratio(self.dataset_splits)
        elif _TimeSeriesSplit and isinstance(self.dataset_splits, _TimeSeriesSplit):
            dataset_periods = self._split_by_cross_validator(self.dataset_splits)
        else:
            # only nudge about installing sklearn when it's actually the missing piece
            sklearn_hint = (
                " (install scikit-learn to use cross-validation splitters)"
                if _TimeSeriesSplit is None
                else ""
            )
            raise ValueError(
                f"`dataset_splits` must be an int, a dict, or a `sklearn.model_selection.TimeSeriesSplit`, but got {type(self.dataset_splits)}{sklearn_hint}"
            )
        # NOTE: use object.__setattr__ as a workaround of (frozen=True) to allow reassignment of derived fields
        object.__setattr__(self, "dataset_periods", dataset_periods)

    @staticmethod
    def _to_ratios(
        dataset_splits: int | DatasetSplitsDict,
    ) -> tuple[float, float, float]:
        """Normalize either input format into a (train, dev, test) ratio triple."""
        if isinstance(dataset_splits, int):
            digits = [int(d) for d in str(dataset_splits)]
            if len(digits) != 3:
                raise ValueError(
                    f'`dataset_splits` int must be 3 digits, e.g. "721" means 70% train, 20% dev, 10% test, but got {dataset_splits}'
                )
            # a 3-digit int is always 100-999, so sum(digits) >= 1 (no zero-division risk)
            total = sum(digits)
            return digits[0] / total, digits[1] / total, digits[2] / total
        elif isinstance(dataset_splits, dict):
            return DatasetSplitter._normalize_split_ratios(dataset_splits)
        else:
            raise TypeError(f"unexpected `dataset_splits` type: {type(dataset_splits)}")

    @staticmethod
    def _normalize_split_ratios(
        splits: DatasetSplitsDict,
    ) -> tuple[float, float, float]:
        all_ratios = {
            "train": splits.get("train"),
            "dev": splits.get("dev"),
            "test": splits.get("test"),
        }
        provided = {k: v for k, v in all_ratios.items() if v is not None}
        if len(provided) < 2:
            raise ValueError(
                f"`dataset_splits` dict must provide at least 2 of 'train'/'dev'/'test' (the third is derived), but got {sorted(provided)}"
            )
        if any(v < 0 for v in provided.values()):
            raise ValueError(
                f"`dataset_splits` ratios must be non-negative, but got {provided}"
            )
        if len(provided) == 2:
            missing = next(k for k in ("train", "dev", "test") if k not in provided)
            remainder = 1.0 - sum(provided.values())
            if remainder < 0.0:
                raise ValueError(
                    f"`dataset_splits` ratios sum to more than 1.0, cannot derive '{missing}' from {provided}"
                )
            provided[missing] = remainder
        elif not math.isclose(sum(provided.values()), 1.0):
            raise ValueError(
                f"`dataset_splits` ratios must sum to 1.0, but got {provided} summing to {sum(provided.values())}"
            )
        return provided["train"], provided["dev"], provided["test"]

    def _split_by_ratio(
        self, dataset_splits: int | DatasetSplitsDict
    ) -> DatasetPeriods:
        total_days = (self.dataset_end - self.dataset_start).days + 1
        train_ratio, dev_ratio, _ = self._to_ratios(dataset_splits)

        # cumulative-boundary rounding: because cumulative ratios are non-decreasing
        # and round() is monotonic, the three segments are guaranteed non-negative and
        # tile [dataset_start, dataset_end] EXACTLY. (Rounding each ratio independently
        # could overflow, pushing dev/test past dataset_end and making test_days < 0.)
        train_days = round(train_ratio * total_days)
        dev_days = round((train_ratio + dev_ratio) * total_days) - train_days
        test_days = total_days - train_days - dev_days

        if train_days <= 0:
            raise ValueError(
                f"resulting train set is empty (train ratio {train_ratio:.3f} over {total_days} days); use a larger `data_range` or a higher train ratio"
            )

        # lay the segments out contiguously; a 0-day dev/test collapses to (None, None)
        cursor = self.dataset_start

        def take(days: int) -> tuple[datetime.date | None, datetime.date | None]:
            nonlocal cursor
            if days <= 0:
                return (None, None)
            segment = (cursor, cursor + datetime.timedelta(days=days - 1))
            cursor = segment[1] + datetime.timedelta(days=1)
            return segment

        train_start = self.dataset_start
        train_end = train_start + datetime.timedelta(days=train_days - 1)
        cursor = train_end + datetime.timedelta(days=1)

        dataset_periods: DatasetPeriods = {
            "dataset": (self.dataset_start, self.dataset_end),
            "train_set": (train_start, train_end),
            "dev_set": take(dev_days),
            "test_set": take(test_days),
        }
        return dataset_periods

    def _split_by_cross_validator(
        self, cross_validator: CrossValidator
    ) -> list[CrossValidatorDatasetPeriods]:
        # fold the whole dataset range; a final hold-out (if wanted) is expressed by
        # narrowing `data_range`, not baked in here (see cv/base.py).
        from pfund._backtest.cv.base import fold_cv_region

        return fold_cv_region(self.dataset_start, self.dataset_end, cross_validator)
