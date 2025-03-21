from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from sklearn.model_selection._split import BaseCrossValidator
    from narwhals.typing import FrameT
    from pfund.typing import DataRangeDict, DatasetSplitsDict

import datetime

from pfeed.utils.utils import rollback_date_range


class Dataset:
    def __init__(self, data_range: str | DataRangeDict, dataset_splits: int | DatasetSplitsDict | BaseCrossValidator=721):
        self.start_date, self.end_date = self._parse_data_range(data_range)
        self.split_ratio, self.cross_validator, self.dataset_splits = self._parse_dataset_splits(dataset_splits)
        
        # TODO: handle cross_validator
        if self.cross_validator:
            raise NotImplementedError('cross_validator is not supported for now')
        
        # TODO
        # self.train_period, self.val_period, self.test_period = self._split_periods()
        self.train_set = self.val_set = self.test_set = None
        
    def _parse_data_range(self, data_range: str | DataRangeDict) -> tuple[datetime.date, datetime.date]:
        if isinstance(data_range, str):
            rollback_period = data_range
            start_date, end_date = rollback_date_range(rollback_period)
        else:
            start_date = datetime.datetime.strptime(data_range['start_date'], '%Y-%m-%d').date()
            if 'end_date' not in data_range:
                yesterday = datetime.datetime.now(tz=datetime.timezone.utc).date() - datetime.timedelta(days=1)
                end_date = yesterday
            else:
                end_date = datetime.datetime.strptime(data_range['end_date'], '%Y-%m-%d').date()
        assert start_date <= end_date, f"start_date must be before end_date: {start_date} <= {end_date}"
        return start_date, end_date
    
    def _parse_dataset_splits(
        self, 
        dataset_splits_input: int | DatasetSplitsDict | BaseCrossValidator
    ) -> tuple[int | None, BaseCrossValidator | None, DatasetSplitsDict | None]:
        split_ratio: int | None = None
        cross_validator: BaseCrossValidator | None = None
        dataset_splits: DatasetSplitsDict | None = None
        if isinstance(dataset_splits_input, int):
            split_ratio = dataset_splits_input
        elif isinstance(dataset_splits_input, BaseCrossValidator):
            cross_validator = dataset_splits_input
        else:
            assert isinstance(dataset_splits_input, dict), f'dataset_splits is expected to be a dict but got {type(dataset_splits_input)}'
            dataset_splits = dataset_splits_input
        return split_ratio, cross_validator, dataset_splits
    
    # TODO:
    def _split_periods(self) -> tuple[tuple[datetime.date, datetime.date], tuple[datetime.date, datetime.date], tuple[datetime.date, datetime.date]]:
        pass
    
    # TODO:
    def _split_datasets(self, df: FrameT) -> tuple[FrameT, FrameT, FrameT]:
        pass