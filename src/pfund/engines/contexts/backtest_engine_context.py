from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal, TypeAlias, TypedDict

if TYPE_CHECKING:
    from pfeed.storages.storage_config import StorageConfig

    from pfund.engines.base_engine import DataRangeDict

from pfund_kit.style import RichColor, TextStyle, cprint

from pfund.datas.resolution import Resolution
from pfund.enums import Environment, BacktestMode
from pfund.engines.settings.backtest_engine_settings import BacktestEngineSettings
from pfund.engines.contexts.base_engine_context import BaseEngineContext
from pfund._backtest.cv.base import CrossValidator
from pfund._backtest.cv.holdout import Holdout


class DatasetSplitsDict(TypedDict, total=False):
    train: float
    val: float
    dev: float
    test: float


DatasetSplits: TypeAlias = int | DatasetSplitsDict | Holdout | CrossValidator | None


class BacktestEngineContext(BaseEngineContext[BacktestEngineSettings]):
    def __init__(
        self,
        env: Environment,
        name: str,
        data_range: str | Resolution | DataRangeDict | tuple[str, str] | Literal["ytd"],
        settings: BacktestEngineSettings | None = None,
        storage_config: StorageConfig | None = None,
        mode: BacktestMode
        | Literal["vectorized", "event_driven"] = BacktestMode.VECTORIZED,
        dataset_splits: DatasetSplits = 721,
    ):
        super().__init__(
            env=env,
            name=name,
            data_range=data_range,
            settings=settings,
            storage_config=storage_config,
        )
        self.mode = BacktestMode[mode.upper()]
        if self.mode == BacktestMode.EVENT_DRIVEN and self.settings.reuse_signals:
            cprint(
                "Warning: Reusing pre-computed features to speed up event-driven backtesting,\n"
                + "i.e. computing features on the fly will be skipped",
                style=TextStyle.BOLD + RichColor.YELLOW,
            )
        self.dataset_splits = self._normalize_dataset_splits(dataset_splits)

    @classmethod
    def _normalize_dataset_splits(
        cls,
        dataset_splits: DatasetSplits,
    ) -> Holdout | CrossValidator | None:
        """Normalize legacy inputs without resolving them against a dataset."""
        if dataset_splits is None or isinstance(
            dataset_splits, (Holdout, CrossValidator)
        ):
            return dataset_splits
        if isinstance(dataset_splits, bool):
            raise TypeError("`dataset_splits` must not be a bool")
        if isinstance(dataset_splits, int):
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
        if isinstance(dataset_splits, dict):
            train, val, test = cls._normalize_split_ratios(dataset_splits)
            return Holdout(train=train, val=val, test=test)
        raise TypeError(
            "`dataset_splits` must be None, an int, a dict, Holdout, or an "
            + f"sklearn-compatible cross-validator, got {type(dataset_splits)}"
        )

    @staticmethod
    def _normalize_split_ratios(
        splits: DatasetSplitsDict,
    ) -> tuple[float, float, float]:
        allowed_keys = {"train", "val", "dev", "test"}
        unknown_keys = set(splits) - allowed_keys
        if unknown_keys:
            raise ValueError(
                "`dataset_splits` dict contains unsupported key(s): "
                + f"{sorted(unknown_keys)}"
            )
        if "val" in splits and "dev" in splits:
            raise ValueError(
                "`dataset_splits` dict cannot provide both 'val' and its legacy "
                + "alias 'dev'"
            )

        all_ratios = {
            "train": splits.get("train"),
            "val": splits.get("val", splits.get("dev")),
            "test": splits.get("test"),
        }
        provided = {
            name: ratio for name, ratio in all_ratios.items() if ratio is not None
        }
        if len(provided) < 2:
            raise ValueError(
                "`dataset_splits` dict must provide at least 2 of "
                + f"'train'/'val'/'test', but got {sorted(provided)}"
            )
        for name, ratio in provided.items():
            if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
                raise TypeError(
                    f"`dataset_splits['{name}']` must be a number, got {type(ratio)}"
                )
            if not math.isfinite(ratio):
                raise ValueError(
                    f"`dataset_splits['{name}']` must be finite, got {ratio}"
                )
            if ratio < 0:
                raise ValueError(
                    f"`dataset_splits` ratios must be non-negative, got {provided}"
                )

        total = sum(provided.values())
        if len(provided) == 2:
            missing = next(
                name for name in ("train", "val", "test") if name not in provided
            )
            remainder = 1.0 - total
            if remainder < 0.0 and not math.isclose(remainder, 0.0, abs_tol=1e-12):
                raise ValueError(
                    "`dataset_splits` ratios sum to more than 1.0, cannot derive "
                    + f"'{missing}' from {provided}"
                )
            provided[missing] = max(0.0, remainder)
        elif not math.isclose(total, 1.0):
            raise ValueError(
                "`dataset_splits` ratios must sum to 1.0, "
                + f"got {provided} summing to {total}"
            )
        return provided["train"], provided["val"], provided["test"]
