from __future__ import annotations
from typing import TypedDict

import datetime
from dataclasses import dataclass, field

try:
    # from sklearn.model_selection._split import BaseCrossValidator
    from sklearn.model_selection import TimeSeriesSplit
except ImportError:
    TimeSeriesSplit = None

from pfund_kit.style import cprint


class DatasetSplitsDict(TypedDict, total=True):
    train: float
    dev: float
    test: float
    

class DatasetPeriods(TypedDict):
    dataset: tuple[datetime.date, datetime.date]
    train_set: tuple[datetime.date, datetime.date]
    dev_set: tuple[datetime.date, datetime.date]
    test_set: tuple[datetime.date, datetime.date]
    
    
class CrossValidatorDatasetPeriods(TypedDict):
    fold: int
    dataset: tuple[datetime.date, datetime.date]
    train_set: tuple[datetime.date, datetime.date]
    dev_set: tuple[datetime.date, datetime.date]
    holdout_test_set: tuple[datetime.date, datetime.date]


@dataclass(frozen=True)
class DatasetSplitter:
    """
    A dataclass that splits a dataset (based on `data_range`) into train/dev/test,
    either by ratio or by TimeSeriesSplit.
    """
    dataset_start: datetime.date
    dataset_end: datetime.date
    dataset_splits: int | DatasetSplitsDict | TimeSeriesSplit = 721  # pyright: ignore[reportInvalidTypeForm]
    cv_test_ratio: float = 0.1

    # Derived fields:
    dataset_periods: DatasetPeriods | list[CrossValidatorDatasetPeriods] = field(init=False)

    def __post_init__(self):
        # NOTE: use object.__setattr__ as a workaround of (frozen=True) to allow reassignment of derived fields
        if isinstance(self.dataset_splits, (int, dict)):
            dataset_periods: DatasetPeriods = self._split_by_ratio()
            object.__setattr__(self, 'dataset_periods', dataset_periods)
        elif TimeSeriesSplit and isinstance(self.dataset_splits, TimeSeriesSplit):
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
            dev_ratio = digits[1] / sum_digits
            # test_ratio = digits[2] / sum_digits
        elif isinstance(self.dataset_splits, dict):
            train_ratio = self.dataset_splits['train']
            dev_ratio = self.dataset_splits['dev']
            # test_ratio = dataset_splits['test']

        train_days = round(train_ratio * total_days)
        dev_days = round(dev_ratio * total_days)
        used_days = train_days + dev_days
        test_days = total_days - used_days
        
        train_start = self.dataset_start
        train_end   = train_start + datetime.timedelta(days=max(0, train_days - 1))
        if dev_days > 0:
            dev_start = train_end + datetime.timedelta(days=1)
            dev_end   = dev_start + datetime.timedelta(days=max(0, dev_days - 1))
            if test_days > 0:
                test_start = dev_end + datetime.timedelta(days=1)
                test_end = self.dataset_end
            else:
                test_start = test_end = None
        else:
            cprint('development/validation set is EMPTY', style='bold')
            dev_start = dev_end = None
            test_start = test_end = None
        
        dataset_periods = {
            'dataset': (self.dataset_start, self.dataset_end),
            'train_set': (train_start, train_end),
            'dev_set': (dev_start, dev_end),
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
        for fold_num, (train_indices, dev_indices) in enumerate(cross_validator.split(range(cv_days))):
            dataset_periods_per_fold.append({
                'fold': fold_num,
                'dataset': (cv_dates[train_indices[0]], cv_dates[dev_indices[-1]]),
                'train_set': (cv_dates[train_indices[0]], cv_dates[train_indices[-1]]),
                'dev_set': (cv_dates[dev_indices[0]], cv_dates[dev_indices[-1]]),
                # NOTE: holdout test set is NOT per fold
                'holdout_test_set': (test_start, test_end),
            })
        return dataset_periods_per_fold
