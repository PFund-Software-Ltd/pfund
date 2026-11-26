from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypedDict

if TYPE_CHECKING:
    from narwhals.typing import IntoDataFrame
    from numpy.typing import NDArray

    class DatasetSplitsDict(TypedDict, total=False):
        train: float
        val: float
        dev: float
        test: float
        # cross-validation folds over everything before the test tail,
        # only combined with `test`, e.g. {"test": 0.1, "folds": WalkForward()}
        folds: CrossValidator

    # the user-facing way to say how to split the dataset:
    # - int: 3 digits read as train/val/test weights, e.g. 721 is 70/20/10.
    # - dict: at least 2 of train/val/test (dev aliases val), a missing third
    #   ratio takes the remainder. Or test + folds: a sealed test tail, then
    #   cross-validation folds over the rest (trainval).
    # - Holdout / CrossValidator: a recipe, passed through as is.
    type DatasetSplitsInput = int | DatasetSplitsDict | Holdout | CrossValidator
    type SplitPart = Literal["train", "val", "trainval", "test"]

import datetime
import math
from dataclasses import dataclass

import narwhals as nw
import numpy as np
from pfund_blocks.resolution import Resolution
from pfund_kit.style import TextStyle, cprint

from pfund.backtest.cv.cross_validation import CrossValidator
from pfund.backtest.cv.holdout import Holdout, _round_half_up


@dataclass(frozen=True)
class Fold:
    """One train/val cut, as positions on the split grid."""

    train: NDArray[np.int64]
    val: NDArray[np.int64]


class DatasetSplits:
    """``DatasetSplitsInput`` resolved once on the engine's data range.

    The data range is laid out as a grid at ``split_resolution``, one sample per
    grid bar. A sealed ``test`` tail is carved first, then ``folds`` are cut over
    everything before it (``trainval``): a holdout is a single fold, a
    cross-validator yields its own folds. Every part is a set of grid positions,
    so it holds whatever a fold needs (gaps, non-contiguous train, ...).

    Components never split themselves; each one maps its own rows onto the grid
    with ``row_indices``. All components share the same boundaries, so a bar
    never lands on different sides of a cut in different components.
    """

    def __init__(
        self,
        dataset_splits: DatasetSplitsInput,
        *,
        data_start: datetime.date,
        data_end: datetime.date,
        split_resolution: str | Resolution,
    ):
        """
        Args:
            dataset_splits: the user's input, see ``DatasetSplitsInput``.
            data_start: first date of the engine's data range.
            data_end: last date of the engine's data range, inclusive.
            split_resolution: bar size of the grid the splits are cut on.
        """
        self.split_resolution = Resolution(split_resolution)
        self.grid: NDArray[np.datetime64] = _make_grid(
            data_start, data_end, self.split_resolution
        )
        self.grid_end = np.datetime64(data_end + datetime.timedelta(days=1), "us")
        test_ratio, recipe = _parse(dataset_splits)
        self.is_holdout = isinstance(recipe, Holdout)
        num_times = len(self.grid)
        if isinstance(recipe, Holdout):
            self.test, self.folds = _cut_holdout(recipe, num_times)
        else:
            self.test, self.folds = _cut_folds(recipe, test_ratio, self.grid)

    @property
    def num_folds(self) -> int:
        return len(self.folds)

    def positions(self, part: SplitPart, fold: int | None = None) -> NDArray[np.int64]:
        """Grid positions of one split part.

        Args:
            part: ``train``/``val`` of a fold, ``trainval`` (everything before
                test) or ``test``.
            fold: required for ``train``/``val`` under cross-validation,
                optional (only 0) under holdout, not allowed otherwise.
        """
        if part in ("trainval", "test"):
            if fold is not None:
                raise ValueError(f"'{part}' does not take a fold, got fold={fold}")
            if part == "test":
                return self.test
            trainval_end = len(self.grid) - len(self.test)
            return np.arange(trainval_end, dtype=np.int64)
        if part not in ("train", "val"):
            raise ValueError(
                f"part must be 'train', 'val', 'trainval' or 'test', got {part!r}"
            )
        if fold is None:
            if not self.is_holdout:
                raise ValueError(
                    f"'{part}' needs fold= under cross-validation, "
                    + f"e.g. fold=0..{self.num_folds - 1}"
                )
            fold = 0
        if not 0 <= fold < self.num_folds:
            raise IndexError(
                f"fold must be in 0..{self.num_folds - 1}, got fold={fold}"
            )
        return getattr(self.folds[fold], part)

    def row_indices(
        self, X: IntoDataFrame, part: SplitPart, fold: int | None = None
    ) -> NDArray[np.int64]:
        """Positions of the rows of ``X`` that fall in one split part.

        Each row belongs to the grid bar its ``date`` lies in.
        """
        dates = nw.from_native(X).get_column("date").to_numpy()
        # NaT makes min()/max() NaT, which would slip past the range check below
        if np.isnat(dates).any():
            raise ValueError("X has rows with a missing date")
        if len(dates) and (dates.min() < self.grid[0] or dates.max() >= self.grid_end):
            raise ValueError("X has rows outside the engine's data range")
        positions = self.positions(part, fold)
        row_bars = np.searchsorted(self.grid, dates, side="right") - 1
        in_part = np.zeros(len(self.grid), dtype=bool)
        in_part[positions] = True
        return np.flatnonzero(in_part[row_bars]).astype(np.int64)


def _parse(
    dataset_splits: DatasetSplitsInput,
) -> tuple[float, Holdout | CrossValidator]:
    """The test ratio and the folds recipe; a holdout carries its own test ratio."""
    if isinstance(dataset_splits, Holdout):
        return dataset_splits.test, dataset_splits
    if isinstance(dataset_splits, CrossValidator):
        return 0.0, dataset_splits
    if isinstance(dataset_splits, bool):
        raise TypeError("`dataset_splits` must not be a bool")
    if isinstance(dataset_splits, int):
        holdout = _parse_int(dataset_splits)
        return holdout.test, holdout
    if isinstance(dataset_splits, dict):
        return _parse_dict(dataset_splits)
    raise TypeError(
        "`dataset_splits` must be an int, a dict, Holdout, or an "
        + f"sklearn-compatible cross-validator, got {type(dataset_splits)}"
    )


def _parse_int(dataset_splits: int) -> Holdout:
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


def _parse_dict(
    splits: DatasetSplitsDict,
) -> tuple[float, Holdout | CrossValidator]:
    """Either test + folds, or holdout ratios (validated by ``Holdout``)."""
    if "folds" in splits:
        unknown_keys = set(splits) - {"test", "folds"}
        if unknown_keys:
            raise ValueError(
                "`dataset_splits` dict with 'folds' only takes 'test', "
                + f"got {sorted(unknown_keys)}"
            )
        test = splits.get("test", 0.0)
        if isinstance(test, bool) or not isinstance(test, (int, float)):
            raise TypeError(f"test ratio must be a number, got {type(test)}")
        if not 0 <= test < 1:
            raise ValueError(f"test ratio must be in [0, 1), got {test}")
        folds = splits["folds"]
        if not isinstance(folds, CrossValidator):
            raise TypeError(
                "'folds' must be an sklearn-compatible cross-validator, "
                + f"got {type(folds)}"
            )
        return float(test), folds

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
    holdout = Holdout(**provided)
    return holdout.test, holdout


def _make_grid(
    data_start: datetime.date, data_end: datetime.date, split_resolution: Resolution
) -> NDArray[np.datetime64]:
    """Bar start times at ``split_resolution`` over the inclusive date range."""
    if not split_resolution.is_bar():
        raise ValueError(
            f"split_resolution must be a bar resolution, got {split_resolution!r}"
        )
    start = np.datetime64(data_start, "us")
    end = np.datetime64(data_end + datetime.timedelta(days=1), "us")
    step = np.timedelta64(split_resolution.to_seconds(), "s")
    grid = np.arange(start, end, step)
    if len(grid) == 0:
        raise ValueError("cannot split an empty data range")
    return grid


def _cut_holdout(
    holdout: Holdout, num_times: int
) -> tuple[NDArray[np.int64], tuple[Fold, ...]]:
    """Ratios are of the whole grid, test is what is left after train and val."""
    train_end = _round_half_up(holdout.train * num_times)
    val_end = _round_half_up((holdout.train + holdout.val) * num_times)
    if train_end <= 0:
        raise ValueError(
            "resulting training set is empty "
            + f"(train ratio {holdout.train:.3f} over {num_times} grid bars); "
            + "use a larger `data_range`, a finer split_resolution or a higher train ratio"
        )
    for name, ratio, size in (
        ("validation", holdout.val, val_end - train_end),
        ("test", holdout.test, num_times - val_end),
    ):
        if ratio > 0 and size <= 0:
            cprint(
                f"{name} set is EMPTY ({name} ratio {ratio:.3f} over {num_times} "
                + "grid bars rounded to 0); use a larger `data_range`, "
                + f"a finer split_resolution or a higher {name} ratio",
                style=TextStyle.BOLD,
            )
    fold = Fold(
        train=np.arange(train_end, dtype=np.int64),
        val=np.arange(train_end, val_end, dtype=np.int64),
    )
    return np.arange(val_end, num_times, dtype=np.int64), (fold,)


def _cut_folds(
    cross_validator: CrossValidator, test_ratio: float, grid: NDArray[np.datetime64]
) -> tuple[NDArray[np.int64], tuple[Fold, ...]]:
    """Carve the test tail, then let the cross-validator fold the rest."""
    num_times = len(grid)
    trainval_end = num_times - _round_half_up(test_ratio * num_times)
    if trainval_end <= 0:
        raise ValueError(
            "nothing is left before the test tail "
            + f"(test ratio {test_ratio:.3f} over {num_times} grid bars)"
        )
    folds = tuple(
        Fold(
            train=_to_positions(train, trainval_end, f"fold {index} train"),
            val=_to_positions(val, trainval_end, f"fold {index} val"),
        )
        for index, (train, val) in enumerate(cross_validator.split(grid[:trainval_end]))
    )
    if not folds:
        raise ValueError("cross-validator produced no folds")
    for index, fold in enumerate(folds):
        if np.intersect1d(fold.train, fold.val).size:
            raise ValueError(f"fold {index}: train and val overlap")
    return np.arange(trainval_end, num_times, dtype=np.int64), folds


def _to_positions(positions: Any, trainval_end: int, name: str) -> NDArray[np.int64]:
    positions = np.asarray(positions)
    if positions.ndim != 1 or positions.size == 0:
        raise ValueError(f"{name} positions must be a non-empty 1-D array")
    if not np.issubdtype(positions.dtype, np.integer):
        raise TypeError(f"{name} positions must be integers")
    if np.any(positions < 0) or np.any(positions >= trainval_end):
        raise IndexError(f"{name} positions fall outside trainval")
    if np.unique(positions).size != positions.size:
        raise ValueError(f"{name} positions must not contain duplicates")
    return np.sort(positions).astype(np.int64)
