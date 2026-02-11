from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund.typing import Currency

from pfund.entities.balances.balance_base import BaseBalance


class CryptoBalance(BaseBalance):
    pass
    