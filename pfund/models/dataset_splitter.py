from __future__ import annotations
from typing_extensions import TypedDict
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund._typing import DatasetSplitsDict

import datetime
from dataclasses import dataclass, field

# from sklearn.model_selection._split import BaseCrossValidator
from sklearn.model_selection import TimeSeriesSplit

from pfund import print_warning


class DatasetPeriods(TypedDict):
    dataset: tuple[datetime.date, datetime.date]
    train_set: tuple[datetime.date, datetime.date]
    val_set: tuple[datetime.date, datetime.date]
    test_set: tuple[datetime.date, datetime.date]
    
    
class CrossValidatorDatasetPeriods(TypedDict):
    fold: int
    dataset: tuple[datetime.date, datetime.date]
    train_set: tuple[datetime.date, datetime.date]
    val_set: tuple[datetime.date, datetime.date]
    holdout_test_set: tuple[datetime.date, datetime.date]


@dataclass(frozen=True)
class DatasetSplitter:
    """
    A dataclass that splits a dataset (based on `data_range`) into train/val/test,
    either by ratio or by TimeSeriesSplit.
    """
    dataset_start: datetime.date
    dataset_end: datetime.date
    dataset_splits: int | DatasetSplitsDict | TimeSeriesSplit = 721
    cv_test_ratio: float = 0.1

    # Derived fields:
    dataset_periods: DatasetPeriods | list[CrossValidatorDatasetPeriods] = field(init=False)

    def __post_init__(self):
        # NOTE: use object.__setattr__ as a workaround of (frozen=True) to allow reassignment of derived fields
        if isinstance(self.dataset_splits, (int, dict)):
            dataset_periods: DatasetPeriods = self._split_by_ratio()
            object.__setattr__(self, 'dataset_periods', dataset_periods)
        elif isinstance(self.dataset_splits, TimeSeriesSplit):
            dataset_periods: list[CrossValidatorDatasetPeriods] = self._split_by_cross_validator()
            object.__setattr__(self, 'dataset_periods', dataset_periods)
        else:
            raise ValueError(f'dataset_splits must be an int, a dict, or a TimeSeriesSplit, but got {type(self.dataset_splits)}')
    
    def _split_by_ratio(self) -> DatasetPeriods:
        total_days = (self.dataset_end - self.dataset_start).days + 1
        if isinstance(self.dataset_splits, int):
            digits = [int(d) for d in str(self.dataset_splits)]
            assert len(digits) == 3, '`dataset_splits` must be a number of length 3, e.g. "721" means 70% train, 20% val, 10% test'
            sum_digits = sum(digits)
            train_ratio = digits[0] / sum_digits
            val_ratio = digits[1] / sum_digits
            # test_ratio = digits[2] / sum_digits
        elif isinstance(self.dataset_splits, dict):
            train_ratio = self.dataset_splits['train']
            val_ratio = self.dataset_splits['val']
            # test_ratio = dataset_splits['test']

        train_days = round(train_ratio * total_days)
        val_days = round(val_ratio * total_days)
        used_days = train_days + val_days
        test_days = total_days - used_days
        
        train_start = self.dataset_start
        train_end   = train_start + datetime.timedelta(days=max(0, train_days - 1))
        if val_days > 0:
            val_start = train_end + datetime.timedelta(days=1)
            val_end   = val_start + datetime.timedelta(days=max(0, val_days - 1))
            if test_days > 0:
                test_start = val_end + datetime.timedelta(days=1)
                test_end = self.dataset_end
            else:
                test_start = test_end = None
        else:
            print_warning('validation set is EMPTY')
            val_start = val_end = None
            test_start = test_end = None
        
        dataset_periods = {
            'dataset': (self.dataset_start, self.dataset_end),
            'train_set': (train_start, train_end),
            'val_set': (val_start, val_end),
            'test_set': (test_start, test_end),
        }
        return dataset_periods

    # TODO: support Purged KFold and Combinatorial Purged KFold?
    def _split_by_cross_validator(self) -> list[CrossValidatorDatasetPeriods]:
        from pandas import date_range
        total_days = (self.dataset_end - self.dataset_start).days + 1
        test_days = round(self.cv_test_ratio * total_days)
        cv_days = total_days - test_days

        cv_start = self.dataset_start
        cv_end = cv_start + datetime.timedelta(days=max(0, cv_days - 1))
        if test_days > 0:
            test_start = cv_end + datetime.timedelta(days=1)
            test_end = self.dataset_end
        else:
            test_start = test_end = None

        cv_dates = date_range(start=cv_start, end=cv_end).date.tolist()
        dataset_periods_per_fold = []
        cross_validator = self.dataset_splits
        for fold_num, (train_indices, val_indices) in enumerate(cross_validator.split(range(cv_days))):
            dataset_periods_per_fold.append({
                'fold': fold_num,
                'dataset': (cv_dates[train_indices[0]], cv_dates[val_indices[-1]]),
                'train_set': (cv_dates[train_indices[0]], cv_dates[train_indices[-1]]),
                'val_set': (cv_dates[val_indices[0]], cv_dates[val_indices[-1]]),
                # NOTE: holdout test set is NOT per fold
                'holdout_test_set': (test_start, test_end),
            })
        return dataset_periods_per_fold
