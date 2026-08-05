from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Holdout:
    """A single chronological train/validation/test split."""

    train: float = 0.7
    val: float = 0.2
    test: float = 0.1

    def __post_init__(self) -> None:
        import math

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
