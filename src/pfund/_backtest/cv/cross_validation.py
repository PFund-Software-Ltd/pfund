from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, cast

from pfund._backtest.cv.base import CrossValidator


@dataclass(frozen=True)
class CrossValidation(CrossValidator):
    """Expanding-window cross-validation for ordered time-series samples.

    The arguments and split behavior follow sklearn's ``TimeSeriesSplit``.
    PFund resolves the resulting timestamp selectors against the component's
    materialized feature dataframe, keeping every row at one timestamp in the
    same fold. Sklearn's per-fold ``test_size`` is exposed by PFund as the
    validation set; cross-validation does not create a final test holdout.
    """

    n_splits: int = 5
    max_train_size: int | None = None
    test_size: int | None = None
    gap: int = 0

    def __post_init__(self) -> None:
        self._validate_positive_int("n_splits", self.n_splits, minimum=2)
        if self.max_train_size is not None:
            self._validate_positive_int("max_train_size", self.max_train_size)
        if self.test_size is not None:
            self._validate_positive_int("test_size", self.test_size)
        self._validate_positive_int("gap", self.gap, minimum=0)

    @staticmethod
    def _validate_positive_int(name: str, value: int, *, minimum: int = 1) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an int, got {type(value)}")
        if value < minimum:
            raise ValueError(f"{name} must be at least {minimum}, got {value}")

    def _make_splitter(self) -> CrossValidator:
        try:
            from sklearn.model_selection import TimeSeriesSplit
        except ImportError as exc:
            raise ImportError(
                "scikit-learn is required to use CrossValidation; "
                + "install pfund with the 'sklearn' extra"
            ) from exc

        return cast(
            "CrossValidator",
            TimeSeriesSplit(
                n_splits=self.n_splits,
                max_train_size=self.max_train_size,
                test_size=self.test_size,
                gap=self.gap,
            ),
        )

    def split(
        self,
        X: object,
        y: object | None = None,
        groups: object | None = None,
    ) -> Iterator[tuple[Any, Any]]:
        return self._make_splitter().split(X, y, groups)

    def get_n_splits(
        self,
        X: object | None = None,
        y: object | None = None,
        groups: object | None = None,
    ) -> int:
        _ = X, y, groups
        return self.n_splits
