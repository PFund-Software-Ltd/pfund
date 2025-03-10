from pydantic import BaseModel


class QuoteDataKwargs(BaseModel):
    orderbook_depth: int=1


class TickDataKwargs(BaseModel):
    pass


class BarDataKwargs(BaseModel):
    skip_first_bar: bool=True
    shifts: dict[str, int] | None=None
