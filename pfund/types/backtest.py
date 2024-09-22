from typing_extensions import TypedDict

from pfeed.types.common_literals import tSUPPORTED_DATA_FEEDS


class BacktestKwargs(TypedDict, total=False):
    data_source: tSUPPORTED_DATA_FEEDS
    start_date: str
    end_date: str
    rollback_period: str
