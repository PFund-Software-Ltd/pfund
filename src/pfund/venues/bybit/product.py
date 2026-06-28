# pyright: reportAttributeAccessIssue=false
from __future__ import annotations
from typing import cast

from enum import StrEnum
from datetime import date

from pfund.entities import BaseProduct
from pfund.enums import AssetTypeModifier, CryptoAssetType, OptionType


class BybitProduct(BaseProduct):
    class Category(StrEnum):
        LINEAR = "LINEAR"
        INVERSE = "INVERSE"
        SPOT = "SPOT"
        OPTION = "OPTION"

    def _derive_product_category(self):
        if self.asset_type == CryptoAssetType.CRYPTO:
            category = self.Category.SPOT
        elif self.is_inverse():
            category = self.Category.INVERSE
        elif self.asset_type == CryptoAssetType.OPT:
            category = self.Category.OPTION
        else:
            category = self.Category.LINEAR
        return category

    @property
    def category(self) -> Category:
        return self._derive_product_category()

    def _create_name(self) -> str:
        if self.is_crypto() or self.is_perpetual():
            # NOTE: spots and perpetuals have duplicated symbols, e.g. BTCUSDT, use basis instead to make them unique
            return "_".join([str(self.source), str(self.basis)])
        else:
            return super()._create_name()

    def _create_symbol(self):
        from pfund.venues.bybit import Bybit

        ebase_asset = Bybit.adapter(self.base_asset, group="assets")
        equote_asset = Bybit.adapter(self.quote_asset, group="assets")
        if self.asset_type == CryptoAssetType.PERP:
            if equote_asset == "USDC":
                symbol = ebase_asset + "PERP"
            else:
                symbol = ebase_asset + equote_asset
        elif self.asset_type == AssetTypeModifier.INV + "-" + CryptoAssetType.PERP:
            assert equote_asset == "USD", (
                f"Only USD-denominated inverse perpetual contracts are supported. Did you mean {self.base_asset}_USD_{self.asset_type}?"
            )
            symbol = ebase_asset + equote_asset
        elif self.asset_type == CryptoAssetType.CRYPTO:
            symbol = ebase_asset + equote_asset
        elif self.asset_type == CryptoAssetType.FUT:
            expiration = cast(date, self.expiration)
            # symbol = e.g. BTC-13DEC24
            if equote_asset == "USDC":
                expiration = expiration.strftime("%d%b%y").upper()
                symbol = "-".join([ebase_asset, expiration])
            # symbol = e.g. BTCUSDT-22AUG25
            elif equote_asset == "USDT":
                expiration = expiration.strftime("%d%b%y").upper()
                symbol = "-".join([ebase_asset + equote_asset, expiration])
            else:
                raise ValueError(
                    f"Only USDC and USDT are supported, not {equote_asset}"
                )
        elif self.asset_type == AssetTypeModifier.INV + "-" + CryptoAssetType.FUT:
            # symbol = e.g. BTCUSDH25
            assert equote_asset == "USD", (
                f"Only USD-denominated inverse futures are supported. Did you mean {self.base_asset}_USD_{self.asset_type}?"
            )
            symbol = ebase_asset + equote_asset + self.contract_code
        elif self.asset_type == CryptoAssetType.OPT:
            expiration = cast(date, self.expiration)
            expiration = expiration.strftime("%d%b%y")
            option_type = cast(OptionType, self.option_type)[0]
            strike_price = str(self.strike_price)
            symbol = "-".join([ebase_asset, expiration, strike_price, option_type])
        else:
            raise ValueError(f"Invalid asset type: {self.asset_type}")
        return symbol
