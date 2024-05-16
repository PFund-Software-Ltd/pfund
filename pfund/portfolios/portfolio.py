from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund.positions.position_base import BasePosition
    from pfund.balances.balance_base import BaseBalance


class Portfolio:
    def __init__(self, positions, balances):
        self.stocks = {}  # {account: {pdt: position}}
        self.futures = {}
        self.options = {}

    def add_position(self, position: BasePosition):
        pass
    
    def add_balance(self, balance: BaseBalance):
        pass
    
    # TODO: add more functionalities, e.g.
    # - get_total_exposure(unit='USD')
    # - get_positions_by_exposure()
    # - get_exposures_by_asset_class()
    # - ...
    
    
    