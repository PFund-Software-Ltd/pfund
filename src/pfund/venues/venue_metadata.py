from __future__ import annotations
from typing import TypeAlias

from pydantic import BaseModel, Field

from pfund.enums import AllAssetType
from pfund.datas.timeframe import Timeframe


ProductCategory: TypeAlias = str


class VenueMetadata(BaseModel):
    asset_types: list[AllAssetType | str]
    stream_resolution_periods: (
        dict[Timeframe, list[int]] | dict[ProductCategory, dict[Timeframe, list[int]]]
    ) = Field(
        description="streaming data resolutions supported by the venue's API (e.g. websocket API)"
    )
    stream_orderbook_levels: list[int] | dict[ProductCategory, list[int]] = Field(
        description="Orderbook levels supported by the venue's API (e.g. websocket API)"
    )
    requires_asyncio_loop: bool = Field(
        default=False,
        description="""
        Whether the venue needs a dedicated asyncio event loop in its own thread.
        True both for native async I/O and for blocking sync REST (e.g. Alpaca)
        that must be offloaded via loop.run_in_executor to avoid
        blocking the engine.
        False only for non-blocking sync APIs (e.g. IBKR's socket),
        which can be called directly and need no loop.
        """,
    )
    support_place_batch_orders: bool = False
    support_cancel_batch_orders: bool = False
    support_amend_batch_orders: bool = False
