from __future__ import annotations
from typing import Any, TypeAlias, Literal

from pydantic import BaseModel, Field, GetCoreSchemaHandler
from pydantic_core import core_schema

from pfund.enums import AllAssetType
from pfund.datas.resolution import Resolution
from pfund.datas.timeframe import Timeframe


ProductCategory: TypeAlias = str
ResolutionPeriod: TypeAlias = int


class _All:
    def __contains__(self, item: Any):
        return isinstance(item, int)

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        # not a validatable/serializable value — just a sentinel; validate by identity of type
        return core_schema.is_instance_schema(cls)


class VenueMetadata(BaseModel):
    has_markets: bool = Field(
        default=False,
        description="Whether the venue supports get_markets(), which returns a list of all available markets",
    )
    asset_types: list[AllAssetType | str]
    supported_resolutions: (
        dict[Resolution | Timeframe, list[ResolutionPeriod] | _All]
        | dict[
            ProductCategory, dict[Resolution | Timeframe, list[ResolutionPeriod] | _All]
        ]
    ) = Field(
        description=(
            "streaming resolutions supported by the venue's API. Absent key = not supported. "
            + "Value = supported periods for that resolution, or _All for any period."
        )
    )
    support_place_batch_orders: bool = False
    support_cancel_batch_orders: bool = False
    support_amend_batch_orders: bool = False
