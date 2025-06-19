from typing import Any

from pydantic import BaseModel, ConfigDict

from pfund.enums.asset_type import AssetTypeModifier, AllAssetType, ASSET_TYPE_ALIASES


class ProductAssetType(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)
    
    as_string: str  # e.g. 'STOCK'/'STK', 'INVERSE-FUTURE'/'IFUT', 'CRYPTO' etc.
    as_tuple: tuple[AssetTypeModifier | AllAssetType, ...] | None = None

    @staticmethod
    def _parse_asset_type_string_to_tuple(asset_type_string: str) -> tuple[AssetTypeModifier | AllAssetType, ...]:
        '''
        Convert asset type string (e.g. 'INVERSE-FUTURE') to tuple of AssetTypeModifier and AssetType (e.g. (AssetTypeModifier.INVERSE, AssetType.FUTURE)).
        '''
        asset_types_and_modifiers = asset_type_string.split('-')
        for i, atm in enumerate(asset_types_and_modifiers):
            if atm in AssetTypeModifier.__members__:
                asset_types_and_modifiers[i] = AssetTypeModifier[atm]
            elif atm in AllAssetType.__members__:
                asset_types_and_modifiers[i] = AllAssetType[atm]
            else:
                raise ValueError(f"Invalid asset type: {atm}")
        return tuple(asset_types_and_modifiers)
    
    def model_post_init(self, __context: Any):
        asset_type_tuple: tuple[AssetTypeModifier | AllAssetType, ...] = self._parse_asset_type_string_to_tuple(self.as_string)
        # Required for frozen=True models
        object.__setattr__(self, 'as_tuple', asset_type_tuple)
        object.__setattr__(self, 'as_string', '-'.join(asset_type_tuple))
    
    def __eq__(self, other: Any):
        if isinstance(other, str):
            other_tuple = self._parse_asset_type_string_to_tuple(other)
            return self.as_tuple == other_tuple
        elif isinstance(other, ProductAssetType):
            return self.as_tuple == other.as_tuple
        return False
    
    def __contains__(self, item: Any):
        return item in self.as_tuple

    def __iter__(self):
        return iter(self.as_tuple)

    def __len__(self):
        return len(self.as_tuple)

    def __str__(self):
        return self.as_string
    
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
    

class ProductBasis(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)
    
    basis: str
    base_asset: str | None = None
    quote_asset: str | None = None
    asset_type: ProductAssetType | None = None
    
    @staticmethod
    def _standardize_asset_type_string(asset_type: str) -> str:
        asset_types_and_modifiers = asset_type.split('-')
        standardized_asset_types_and_modifiers = [ASSET_TYPE_ALIASES.get(atm.upper(), atm) for atm in asset_types_and_modifiers]
        standardized_asset_type = '-'.join(standardized_asset_types_and_modifiers)
        return standardized_asset_type
    
    def model_post_init(self, __context: Any):
        base_asset, quote_asset, asset_type = self.basis.split('_')
        asset_type = ProductAssetType(as_string=self._standardize_asset_type_string(asset_type))
        object.__setattr__(self, 'base_asset', base_asset)
        object.__setattr__(self, 'quote_asset', quote_asset)
        object.__setattr__(self, 'asset_type', asset_type)  # Required when frozen=True

    @property
    def asset_pair(self) -> str:
        return '_'.join([self.base_asset, self.quote_asset])
    
    def __str__(self):
        return '_'.join([self.asset_pair, self.asset_type.as_string])
