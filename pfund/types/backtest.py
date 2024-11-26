from typing_extensions import TypedDict

from pfeed.types.literals import tDATA_SOURCE, tSTORAGE


class BacktestKwargs(TypedDict, total=False):
    data_source: tDATA_SOURCE
    start_date: str
    end_date: str
    rollback_period: str
    from_storage: tSTORAGE | None