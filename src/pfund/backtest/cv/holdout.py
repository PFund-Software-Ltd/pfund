from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from narwhals.typing import IntoDataFrame
    from numpy.typing import NDArray

import math
from dataclasses import dataclass

import numpy as np
from pfund_kit.style import TextStyle, cprint

from pfund.backtest.cv.holdout_split import HoldoutSplit
from pfund.backtest.cv.indexing import resolve_time_spine


def _round_half_up(value: float) -> int:
    return math.floor(value + 0.5)


@dataclass(frozen=True)
class Holdout:
    """A single chronological train/validation/test split."""

    train: float = 0.7
    val: float = 0.2
    test: float = 0.1

    def __post_init__(self) -> None:
        ratios = {
            "train": self.train,
            "val": self.val,
            "test": self.test,
        }
        for name, ratio in ratios.items():
            if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
                raise TypeError(f"{name} ratio must be a number, got {type(ratio)}")
            if not math.isfinite(ratio):
                raise ValueError(f"{name} ratio must be finite, got {ratio}")
            if ratio < 0:
                raise ValueError(f"{name} ratio must be non-negative, got {ratio}")

        if self.train == 0:
            raise ValueError("train ratio must be greater than 0")
        if not math.isclose(sum(ratios.values()), 1.0):
            raise ValueError(
                "holdout ratios must sum to 1.0, "
                + f"got {ratios} summing to {sum(ratios.values())}"
            )

    @property
    def dev(self) -> float:
        """Alias retained for pfund's existing train/dev/test terminology."""
        return self.val


def resolve_holdout(X: IntoDataFrame, holdout: Holdout) -> HoldoutSplit:
    """Resolve holdout ratios against the actual unique timestamps in ``X``."""
    time_spine, row_time_indices = resolve_time_spine(
        X,
        operation="holdout splitting",
    )
    num_times = len(time_spine)

    train_size = _round_half_up(holdout.train * num_times)
    val_size = _round_half_up((holdout.train + holdout.val) * num_times) - train_size
    test_size = num_times - train_size - val_size

    if train_size <= 0:
        raise ValueError(
            "resulting training set is empty "
            + f"(train ratio {holdout.train:.3f} over {num_times} timestamps); "
            + "use a larger `data_range` or a higher train ratio"
        )
    if holdout.val > 0 and val_size <= 0:
        cprint(
            "validation set is EMPTY "
            + f"(val ratio {holdout.val:.3f} over {num_times} timestamps "
            + "rounded to 0); use a larger `data_range` or a higher val ratio",
            style=TextStyle.BOLD,
        )
    if holdout.test > 0 and test_size <= 0:
        cprint(
            "test set is EMPTY "
            + f"(test ratio {holdout.test:.3f} over {num_times} timestamps "
            + "rounded to 0); use a larger `data_range` or a higher test ratio",
            style=TextStyle.BOLD,
        )

    train_end = train_size
    val_end = train_end + val_size

    def take_time_range(start: int, end: int) -> NDArray[np.int64] | None:
        if start == end:
            return None
        return np.flatnonzero(
            (row_time_indices >= start) & (row_time_indices < end)
        ).astype(np.int64, copy=False)

    train_indices = take_time_range(0, train_end)
    assert train_indices is not None
    return HoldoutSplit(
        X,
        indices=(
            train_indices,
            take_time_range(train_end, val_end),
            take_time_range(val_end, num_times),
        ),
    )
