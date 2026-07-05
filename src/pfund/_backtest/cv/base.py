"""
Cross-validation for backtesting.

This module folds a contiguous date range into per-fold train/dev date periods.
`fold_cv_region` handles splitters whose train indices are contiguous per fold
(e.g. `TimeSeriesSplit`), reporting each set as a `(first, last)` date span. The
purged variants in this package yield *non-contiguous* train indices (purging
removes samples mid-set), so they will need their own span handling rather than
reusing `fold_cv_region` as-is.
"""

from __future__ import annotations

import datetime
from collections.abc import Iterator
from typing import Any, Protocol, TypedDict


class CrossValidatorDatasetPeriods(TypedDict):
    fold: int
    dataset: tuple[datetime.date, datetime.date]
    train_set: tuple[datetime.date, datetime.date]
    dev_set: tuple[datetime.date, datetime.date]


class CrossValidator(Protocol):
    """Minimal sklearn-compatible splitter interface, e.g. `sklearn.model_selection.TimeSeriesSplit`."""

    # index arrays are typed Any: sklearn yields numpy ndarrays, and this is an
    # external contract we adapt rather than own.
    def split(
        self, X: object, y: object = None, groups: object = None
    ) -> Iterator[tuple[Any, Any]]: ...


def fold_cv_region(
    start: datetime.date,
    end: datetime.date,
    cross_validator: CrossValidator,
) -> list[CrossValidatorDatasetPeriods]:
    """
    Fold the contiguous [start, end] range (one sample per calendar day) into
    per-fold train/dev date periods.

    NOTE: sklearn yields (train_idx, test_idx) per fold; the fold's "test" split is
    our per-fold *dev* (validation) set.
    """
    cv_dates = [
        start + datetime.timedelta(days=i) for i in range((end - start).days + 1)
    ]
    periods: list[CrossValidatorDatasetPeriods] = []
    for fold_num, (train_indices, dev_indices) in enumerate(
        cross_validator.split(range(len(cv_dates)))
    ):
        periods.append(
            {
                "fold": fold_num,
                "dataset": (cv_dates[train_indices[0]], cv_dates[dev_indices[-1]]),
                "train_set": (cv_dates[train_indices[0]], cv_dates[train_indices[-1]]),
                "dev_set": (cv_dates[dev_indices[0]], cv_dates[dev_indices[-1]]),
            }
        )
    return periods
