from __future__ import annotations
from typing import TYPE_CHECKING, TypeAlias

from pydantic import BaseModel

from pfund.enums import TradingVenue


class RiskGuard(BaseModel):
    pass


class RiskManager:
    def __init__(self):
        pass
