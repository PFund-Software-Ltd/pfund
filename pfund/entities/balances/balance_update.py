from typing import Literal, ClassVar

import time
from decimal import Decimal

from pydantic import BaseModel, Field, PrivateAttr, ConfigDict

from pfund.typing import Currency


class BalanceUpdate(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra='forbid', frozen=True)
    
    _created_at: float = PrivateAttr(default_factory=time.time)
    ts: float | None = Field(default=None, description='timestamp (if any) included in the update sent from the trading venue')
    data: dict[
        Currency, 
        dict[str | Literal['wallet', 'available', 'margin'], Decimal]
    ]

    @property
    def created_at(self) -> float:
        return self._created_at
