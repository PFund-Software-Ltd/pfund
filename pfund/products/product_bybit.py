from __future__ import annotations
from typing import Literal

from enum import StrEnum

from pydantic import model_validator

from pfund.enums import TradingVenue, CryptoExchange, CryptoAssetType, AssetTypeModifier
from pfund.products.product_crypto import CryptoProduct
from pfund.adapter import Adapter


# REVIEW
SUPPORTED_ASSET_TYPES: list = [
    CryptoAssetType.FUTURE,
    CryptoAssetType.PERPETUAL,
    CryptoAssetType.OPTION,
    CryptoAssetType.CRYPTO,
    CryptoAssetType.INDEX,
    AssetTypeModifier.INVERSE + '-' + CryptoAssetType.FUTURE,
    AssetTypeModifier.INVERSE + '-' + CryptoAssetType.PERPETUAL,
]


tPRODUCT_CATEGORY = Literal['LINEAR', 'INVERSE', 'SPOT', 'OPTION']
class ProductCategory(StrEnum):
    LINEAR = 'LINEAR'
    INVERSE = 'INVERSE'
    SPOT = 'SPOT'
    OPTION = 'OPTION'


class BybitProduct(CryptoProduct):
    trading_venue: TradingVenue = TradingVenue.BYBIT
    exchange: CryptoExchange = CryptoExchange.BYBIT
    adapter: Adapter | None = None
    category: ProductCategory | None = None

    @model_validator(mode='after')
    def _derive_product_category(self) -> ProductCategory:
        if self.asset_type == CryptoAssetType.CRYPTO:
            category = ProductCategory.SPOT
        elif self.is_inverse():
            category = ProductCategory.INVERSE
        elif self.asset_type == CryptoAssetType.OPT:
            category = ProductCategory.OPTION
        else:
            category = ProductCategory.LINEAR
        self.category = category
        return self
    
    @model_validator(mode='after')
    def _validate_asset_type(self):
        if str(self.asset_type) not in SUPPORTED_ASSET_TYPES:
            raise ValueError(f"Invalid asset type: {self.asset_type}")
        return self
    
    def _create_symbol(self):
        ebase_asset = self.adapter(self.base_asset, group='asset')
        equote_asset = self.adapter(self.quote_asset, group='asset')
        if self.asset_type == CryptoAssetType.PERP:
            if equote_asset == 'USDC':
                symbol = ebase_asset + str(self.asset_type)
            else:
                symbol = ebase_asset + equote_asset
        elif self.asset_type == AssetTypeModifier.INVERSE + '-' + CryptoAssetType.PERP:
            assert equote_asset == 'USD', 'only USD-denominated inverse perpetual contracts are supported'
            symbol = ebase_asset + equote_asset
        elif self.asset_type == CryptoAssetType.CRYPTO:
            symbol = ebase_asset + equote_asset
        elif self.asset_type == CryptoAssetType.FUT:
            # symbol = e.g. BTC-13DEC24
            expiration = self.expiration.strftime("%d%b%y")
            symbol = '-'.join([ebase_asset, expiration])
        elif self.asset_type == AssetTypeModifier.INVERSE + '-' + CryptoAssetType.FUT:
            # symbol = e.g. BTCUSDH25
            assert equote_asset == 'USD', 'only USD-denominated inverse perpetual contracts are supported'
            symbol = ebase_asset + equote_asset + self.contract_code
        elif self.asset_type == CryptoAssetType.OPT:
            expiration = self.expiration.strftime("%d%b%y")
            option_type = self.option_type[0]
            strike_price = str(self.strike_price)
            symbol = '-'.join([ebase_asset, expiration, strike_price, option_type])
        return symbol
