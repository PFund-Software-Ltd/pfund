from typing import Any

from functools import cached_property

from pydantic import BaseModel, ConfigDict, field_validator

from pfund.enums.asset_type import AssetTypeModifier, AllAssetType, TraditionalAssetType, CryptoAssetType, DeFiAssetType


class AssetType(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True, ignored_types=(cached_property,))

    value: str  # e.g. 'STOCK'/'STK', 'INVERSE-FUTURE'/'IFUT', 'CRYPTO' etc.

    @staticmethod
    def _parse_asset_type_string_to_tuple(asset_type_string: str) -> tuple[AssetTypeModifier | AllAssetType, ...]:
        '''
        Convert asset type string (e.g. 'INVERSE-FUTURE') to tuple of AssetTypeModifier and AssetType (e.g. (AssetTypeModifier.INVERSE, AssetType.FUTURE)).
        '''
        asset_types_and_modifiers = asset_type_string.split('-')
        for i, atm in enumerate(asset_types_and_modifiers):
            if atm in AssetTypeModifier.__members__:
                asset_types_and_modifiers[i] = AssetTypeModifier[atm]
            # NOTE: some aliases are not in AllAssetType, e.g. 'SPOT' only exists in CryptoAssetType
            elif atm in AllAssetType.__members__:
                asset_types_and_modifiers[i] = AllAssetType[atm]
            elif atm in TraditionalAssetType.__members__:
                asset_types_and_modifiers[i] = TraditionalAssetType[atm]
            elif atm in CryptoAssetType.__members__:
                asset_types_and_modifiers[i] = CryptoAssetType[atm]
            elif atm in DeFiAssetType.__members__:
                asset_types_and_modifiers[i] = DeFiAssetType[atm]
            else:
                raise ValueError(f"Invalid asset type: {atm}")
        return tuple(asset_types_and_modifiers)

    @field_validator('value', mode='after')
    @classmethod
    def _normalize(cls, v: str) -> str:
        return '-'.join(cls._parse_asset_type_string_to_tuple(v))

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
