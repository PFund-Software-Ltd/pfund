from __future__ import annotations
from typing import TYPE_CHECKING, TypeAlias
if TYPE_CHECKING:
    from pfeed.enums import DataSource
    from pfund.datas.resolution import Resolution
    from pfund.products.product_base import BaseProduct

from pfund.datas.data_market import MarketData

Price: TypeAlias = float
Size: TypeAlias = float


class QuoteData(MarketData):
    def __init__(
        self,
        data_source: DataSource,
        data_origin: str,
        product: BaseProduct,
        resolution: Resolution
    ):
        super().__init__(data_source, data_origin, product, resolution)
        self._orderbook_depth: int = resolution.period
        self._orderbook_level: int = resolution.orderbook_level
        assert self._orderbook_level in [1, 2, 3]
        try:
            from order_book import OrderBook as FastOrderBook
            self._orderbook = FastOrderBook()
            self._is_fast_orderbook = True
        except ImportError:
            from pfund.datas.orderbook import OrderBook
            self._orderbook = OrderBook()
            self._is_fast_orderbook = False
    
    @property
    def orderbook(self):
        return self._orderbook

    @property
    def level(self) -> int:
        return self._orderbook_level
    
    @property
    def depth(self) -> int:
        return self._orderbook_depth
    
    def get_bid(self, level: int=0) -> tuple[Price, Size]:
        if self._is_fast_orderbook:
            return self._orderbook.bids.index(level)
        else:
            return self._orderbook.get_bid(level)
    
    def get_ask(self, level: int=0) -> tuple[Price, Size]:
        if self._is_fast_orderbook:
            return self._orderbook.asks.index(level)
        else:
            return self._orderbook.get_ask(level)
    
    def get_bid_price(self, level: int=0) -> Price:
        price, _ = self.get_bid(level)
        return price
    
    def get_ask_price(self, level: int=0) -> Price:
        price, _ = self.get_ask(level)
        return price
    
    def get_bid_size(self, level: int=0) -> Size:
        _, size = self.get_bid(level)
        return size
    
    def get_ask_size(self, level: int=0) -> Size:
        _, size = self.get_ask(level)
        return size
    
    def on_quote(self, bids, asks, ts: float | None, is_backfill=False, **extra_data):
        # e.g. IB's reqMktData may send only asks with empty bids
        if bids:
            self._orderbook.bids = bids
        if asks:
            self._orderbook.asks = asks
        self.update_ts(ts)
        self.update_extra_data(extra_data)
        
        for resamplee in self._resamplees:
            resamplee.on_quote(bids, asks, ts)

    def __getstate__(self):
        state = self.__dict__.copy()
        # drop the non-picklable C object and flag
        state['_orderbook'] = None
        state['_is_fast_orderbook'] = False
        return state

    def __setstate__(self, state):
        # restore everything else
        self.__dict__.update(state)
        # re-create a pure-Python OrderBook so at least your API still works
        self._orderbook = OrderBook()
        self._is_fast_orderbook = False