from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator

from pfund.entities.products.asset_type import AssetType
from pfund.enums.asset_type import ASSET_TYPE_ALIASES


class ProductBasis(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    basis: str
    base_asset: str | None = None
    quote_asset: str | None = None
    asset_type: AssetType | None = None

    @staticmethod
    def _standardize_asset_type_string(asset_type: str) -> str:
        asset_types_and_modifiers = asset_type.split("-")
        standardized_asset_types_and_modifiers = [
            ASSET_TYPE_ALIASES.get(atm.upper(), atm)
            for atm in asset_types_and_modifiers
        ]
        standardized_asset_type = "-".join(standardized_asset_types_and_modifiers)
        return standardized_asset_type

    @field_validator("basis", mode="before")
    @classmethod
    def validate_product(cls, basis: str) -> str:
        # use regex to validate product string format, it must be like "XXX_YYY_ZZZ" or "XXX_YYY_ZZZ-ZZZ"
        # where the maximum length of each part is 10
        import re

        max_len = 10
        pattern = (
            r"^[A-Za-z0-9.]{1,"
            + str(max_len)
            + "}"
            + r"_[A-Za-z]{1,"
            + str(max_len)
            + "}"
            + r"_[A-Za-z]{1,"
            + str(max_len)
            + "}"
            + r"(?:-[A-Za-z]{1,"
            + str(max_len)
            + "})?$"
        )
        if not re.match(pattern, basis):
            raise ValueError(
                f"Invalid product basis format: `{basis}`. "
                + 'Product basis must be in format "XXX_YYY_ZZZ" or "XXX_YYY_ZZZ-ZZZ" (e.g. "TSLA_USD_STK", "BTC_USDT_SPOT", "ETH_USDT_PERP", "BTC_USD_INVERSE-PERP") where each part contains only letters '
                + f"and maximum {max_len} characters long."
            )
        return basis

    def model_post_init(self, __context: Any):
        base_asset, quote_asset, asset_type = self.basis.split("_")
        asset_type = AssetType(value=self._standardize_asset_type_string(asset_type))
        object.__setattr__(self, "base_asset", base_asset)
        object.__setattr__(self, "quote_asset", quote_asset)
        object.__setattr__(self, "asset_type", asset_type)  # Required when frozen=True

    @property
    def asset_pair(self) -> str:
        return "_".join([self.base_asset, self.quote_asset])

    def __str__(self):
        return "_".join([self.asset_pair, str(self.asset_type)])
