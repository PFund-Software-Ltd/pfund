from functools import cached_property
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from pfund.enums.asset_type import (
    ASSET_TYPE_ALIASES,
    AllAssetType,
    AssetTypeModifier,
    CryptoAssetType,
    TraditionalAssetType,
    PredictionMarketAssetType,
)


class AssetType(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True, frozen=True, ignored_types=(cached_property,)
    )

    value: str  # e.g. 'STOCK'/'STK', 'INVERSE-FUTURE'/'IFUT', 'CRYPTO' etc.

    @model_validator(mode="before")
    @classmethod
    def _coerce_string(cls, data: Any) -> Any:
        # allow a bare string (e.g. "PERP") to be used wherever an
        # AssetType is expected, by treating it as the `value` field
        if isinstance(data, str):
            return {"value": data.upper()}
        return data

    @staticmethod
    def _standardize_asset_type_string(asset_type_string: str) -> str:
        """
        Expand aliases (e.g. 'IPERP' -> 'INVERSE-PERPETUAL') into their canonical hyphenated form.
        """
        asset_types_and_modifiers = asset_type_string.split("-")
        standardized_asset_types_and_modifiers = [
            ASSET_TYPE_ALIASES.get(atm.upper(), atm)
            for atm in asset_types_and_modifiers
        ]
        return "-".join(standardized_asset_types_and_modifiers)

    @staticmethod
    def _parse_asset_type_string_to_tuple(
        asset_type_string: str,
    ) -> tuple[AssetTypeModifier | AllAssetType, ...]:
        """
        Convert asset type string (e.g. 'INVERSE-FUTURE') to tuple of AssetTypeModifier and AssetType (e.g. (AssetTypeModifier.INVERSE, AssetType.FUTURE)).
        """
        asset_type_string = AssetType._standardize_asset_type_string(asset_type_string)
        parsed: list[AssetTypeModifier | AllAssetType] = []
        for atm in asset_type_string.split("-"):
            if atm in AssetTypeModifier.__members__:
                parsed.append(AssetTypeModifier[atm])
            elif atm in AllAssetType.__members__:
                parsed.append(AllAssetType[atm])
            # NOTE: some aliases are not in AllAssetType, e.g. 'SPOT' only exists in CryptoAssetType.
            # The sub-enums' values are AllAssetType members, so normalize to AllAssetType.
            elif atm in TraditionalAssetType.__members__:
                parsed.append(AllAssetType(TraditionalAssetType[atm]))
            elif atm in CryptoAssetType.__members__:
                parsed.append(AllAssetType(CryptoAssetType[atm]))
            elif atm in PredictionMarketAssetType.__members__:
                parsed.append(AllAssetType(PredictionMarketAssetType[atm]))
            else:
                raise ValueError(f"Invalid asset type: {atm}")
        return tuple(parsed)

    @field_validator("value", mode="after")
    @classmethod
    def _normalize(cls, v: str) -> str:
        return "-".join(cls._parse_asset_type_string_to_tuple(v))

    @cached_property
    def as_tuple(self) -> tuple[AssetTypeModifier | AllAssetType, ...]:
        return self._parse_asset_type_string_to_tuple(self.value)

    def __eq__(self, other: Any):
        if isinstance(other, str):
            other_tuple = self._parse_asset_type_string_to_tuple(other)
            return self.as_tuple == other_tuple
        elif isinstance(other, AssetType):
            return self.as_tuple == other.as_tuple
        return False

    def __hash__(self) -> int:
        return hash(self.as_tuple)

    def __contains__(self, item: Any):
        return item in self.as_tuple

    def __iter__(self):
        return iter(self.as_tuple)

    def __len__(self):
        return len(self.as_tuple)

    def __str__(self):
        return self.value

    def is_inverse(self) -> bool:
        return AssetTypeModifier.INVERSE in self.as_tuple

    def is_crypto(self) -> bool:
        return AllAssetType.CRYPTO in self.as_tuple

    def is_future(self) -> bool:
        return AllAssetType.FUTURE in self.as_tuple

    def is_perpetual(self) -> bool:
        return AllAssetType.PERPETUAL in self.as_tuple

    def is_option(self) -> bool:
        return AllAssetType.OPTION in self.as_tuple

    def is_stock(self) -> bool:
        return AllAssetType.STOCK in self.as_tuple

    def is_etf(self) -> bool:
        return AllAssetType.ETF in self.as_tuple

    def is_forex(self) -> bool:
        return AllAssetType.FOREX in self.as_tuple

    def is_bond(self) -> bool:
        return AllAssetType.BOND in self.as_tuple

    def is_mutual_fund(self) -> bool:
        return AllAssetType.FUND in self.as_tuple

    def is_commodity(self) -> bool:
        return AllAssetType.CMDTY in self.as_tuple

    def is_index(self) -> bool:
        return AllAssetType.INDEX in self.as_tuple
