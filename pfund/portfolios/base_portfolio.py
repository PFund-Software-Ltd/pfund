from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund.positions.position_base import BasePosition

    
class BasePortfolio:
    @classmethod
    def from_positions_and_balances(cls, positions, balances):
        return cls(positions, balances)
    
    def _get_positions(self, ptype: str):
        return self._all_positions[ptype.upper()]
    
    def add(self, position: BasePosition):
        positions: 