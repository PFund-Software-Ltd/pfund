from typing_extensions import TypedDict


class QuoteDataKwargs(TypedDict, total=True):
    orderbook_depth: int=1


class TickDataKwargs(TypedDict, total=True):
    pass
    

class BarDataKwargs(TypedDict, total=True):
    skip_first_bar: bool=True
    shifts: dict[str, int] | None=None
