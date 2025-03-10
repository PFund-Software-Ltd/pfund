from __future__ import annotations

import datetime

from pydantic import BaseModel, ConfigDict, model_validator, field_validator

from pfeed.const.enums import DataSource, DataStorage
from pfeed.utils.utils import rollback_date_range


class BacktestKwargs(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    data_source: DataSource
    start_date: datetime.date | None=None
    end_date: datetime.date | None=None
    rollback_period: str | None=None
    from_storage: DataStorage | None=None

    @model_validator(mode='after')
    @classmethod
    def validate_dates_and_rollback_period(cls, data: BacktestKwargs):
        start_date, end_date = data.start_date, data.end_date
        rollback_period = data.rollback_period
        if start_date and not end_date:
            raise ValueError("'end_date' is required when 'start_date' is provided")
        elif end_date and not start_date:
            raise ValueError("'start_date' is required when 'end_date' is provided")
        elif start_date and end_date:
            if start_date > end_date:
                raise ValueError("'start_date' must be before 'end_date'")
            if rollback_period:
                raise ValueError("'rollback_period' is not allowed when 'start_date' and 'end_date' are provided")
        else:
            if not rollback_period:
                raise ValueError("at least one of 'start_date', 'end_date', or 'rollback_period' must be provided")
            else:
                start_date, end_date = rollback_date_range(rollback_period)
                data.start_date = start_date
                data.end_date = end_date
        return data
    
    @field_validator('data_source', mode='before')
    @classmethod
    def validate_data_source(cls, data_source: str) -> str:
        from pfeed.const.aliases import ALIASES
        data_source = data_source.upper()
        SUPPORTED_DATA_SOURCES = [ds.value for ds in DataSource]
        data_source = ALIASES.get(data_source, data_source)
        if data_source not in SUPPORTED_DATA_SOURCES:
            raise ValueError(f"'data_source' must be one of {SUPPORTED_DATA_SOURCES}")
        return data_source
