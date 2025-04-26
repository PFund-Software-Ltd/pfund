from __future__ import annotations
from typing import TYPE_CHECKING, TypeAlias
if TYPE_CHECKING:
    from pfund.datas.resolution import Resolution
    from pfund.products.product_base import BaseProduct

from dataclasses import dataclass, field

from pfund.datas.data_time_based import TimeBasedData


Price: TypeAlias = float
Size: TypeAlias = float


@dataclass
class OrderBook:
    bids: dict[Price, Size] = field(default_factory=dict)
    asks: dict[Price, Size] = field(default_factory=dict)

    def get_bid(self, level: int=0) -> tuple[Price, Size]:
        bid_pxs = sorted(self.bids.keys(), key=lambda px: float(px), reverse=True)
        return bid_pxs[level], self.bids[bid_pxs[level]]
    
    def get_ask(self, level: int=0) -> tuple[Price, Size]:
        ask_pxs = sorted(self.asks.keys(), key=lambda px: float(px), reverse=False)
        return ask_pxs[level], self.asks[ask_pxs[level]]
    

class QuoteData(TimeBasedData):
    def __init__(self, product: BaseProduct, resolution: Resolution, orderbook_depth: int=1, fast_orderbook: bool=True):
        super().__init__(product, resolution)
        self._orderbook_depth = orderbook_depth
        self._orderbook_level = resolution.orderbook_level
        self._fast_orderbook = fast_orderbook
        if fast_orderbook:
            from order_book import OrderBook as FastOrderBook
            self._orderbook = FastOrderBook()
        else:
            self._orderbook = OrderBook()
        assert 0 < self.period <= 1, f'period {self.period} is not supported for QuoteData'
    
    def get_bid(self, level: int=0) -> tuple[Price, Size]:
        if self._fast_orderbook:
            return self._orderbook.bids.index(level)
        else:
            return self._orderbook.get_bid(level)
    
    def get_ask(self, level: int=0) -> tuple[Price, Size]:
        if self._fast_orderbook:
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
    
    @property
    def orderbook_level(self):
        return self._orderbook_level
    
    @property
    def orderbook_depth(self):
        return self._orderbook_depth
