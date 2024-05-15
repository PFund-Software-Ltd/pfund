from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund.positions.position_base import BasePosition
    from pfund.balances.balance_base import BaseBalance


class Portfolio:
    def __init__(self, positions, balances):
        self.stocks = {}
        self.futures = {}
        self.options = {}

    def add_position(self, position: BasePosition):
        pass
    
    def add_balance(self, balance: BaseBalance):
        pass
    
    
    