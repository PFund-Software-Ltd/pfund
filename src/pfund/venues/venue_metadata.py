from __future__ import annotations
from typing import TYPE_CHECKING, Any, TypeAlias, cast

if TYPE_CHECKING:
    from pfund.entities.products.product_base import BaseProduct

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


SupportedResolutions: TypeAlias = dict[
    Resolution | Timeframe, list[ResolutionPeriod] | _All
]
SupportedResolutionsByCategory: TypeAlias = dict[ProductCategory, SupportedResolutions]


class VenueMetadata(BaseModel):
    has_markets: bool = Field(
        default=False,
        description="Whether the venue supports get_markets(), which returns a list of all available markets",
    )
    asset_types: list[AllAssetType | str]
    supported_resolutions: SupportedResolutions | SupportedResolutionsByCategory = (
        Field(
            description=(
                "streaming resolutions supported by the venue's API. Absent key = not supported. "
                + "Value = supported periods for that resolution, or _All for any period."
            )
        )
    )
    support_place_batch_orders: bool = False
    support_cancel_batch_orders: bool = False
    support_amend_batch_orders: bool = False

    def get_supported_resolutions(self, product: BaseProduct) -> SupportedResolutions:
        supported_resolutions = self.supported_resolutions
        if any(isinstance(value, dict) for value in supported_resolutions.values()):
            resolutions_by_category = cast(
                "SupportedResolutionsByCategory", supported_resolutions
            )
            category = cast(
                "ProductCategory | None", getattr(product, "category", None)
            )
            if category is None or category not in resolutions_by_category:
                raise ValueError(f"{product.desc_str()} has unsupported {category=}")
            return resolutions_by_category[category]
        return cast("SupportedResolutions", supported_resolutions)
