from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund.datas.data_quote import Price, Size

from dataclasses import dataclass, field


# OPTIMIZE: use Rust to create an orderbook class that can handle level-2 operations: insert, delete, update
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