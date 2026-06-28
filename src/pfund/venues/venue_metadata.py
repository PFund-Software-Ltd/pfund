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
    support_place_batch_orders: bool = False
    support_cancel_batch_orders: bool = False
    support_amend_batch_orders: bool = False
