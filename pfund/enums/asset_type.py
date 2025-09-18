from enum import StrEnum


class AssetTypeModifier(StrEnum):
    INVERSE = INV = 'INVERSE'
    
    @property
    def Mixin(self):
        from pfund.products.mixins.inverse import InverseMixin
        return {
            AssetTypeModifier.INVERSE: InverseMixin,
        }[self]


class AllAssetType(StrEnum):
    STOCK = STK = 'STOCK'
    FUTURE = FUT = 'FUTURE'
    PERPETUAL = PERP = 'PERPETUAL'
    OPTION = OPT = 'OPTION'
    FOREX = FX = 'FOREX'
    CRYPTOCURRENCY = CRYPTO = 'CRYPTO'
    COMMODITY = CMDTY = 'COMMODITY'
    ETF = 'ETF'  # exchange-traded fund
    FUND = 'FUND'  # mutual fund
    BOND = 'BOND'
    INDEX = 'INDEX'

    @property
    def Mixin(self):
        from pfund.products.mixins.stock import StockMixin
        from pfund.products.mixins.future import FutureMixin
        from pfund.products.mixins.perpetual import PerpetualMixin
        from pfund.products.mixins.option import OptionMixin
        from pfund.products.mixins.forex import ForexMixin
        from pfund.products.mixins.crypto import CryptoMixin
        from pfund.products.mixins.index import IndexMixin
        return {
            AllAssetType.INDEX: IndexMixin,
            AllAssetType.STOCK: StockMixin,
            AllAssetType.FUTURE: FutureMixin,
            AllAssetType.PERPETUAL: PerpetualMixin,
            AllAssetType.OPTION: OptionMixin,
            AllAssetType.FOREX: ForexMixin,
            AllAssetType.CRYPTO: CryptoMixin,
        }[self]
        

class TraditionalAssetType(StrEnum):
    STOCK = STK = AllAssetType.STOCK
    FUTURE = FUT = AllAssetType.FUTURE
    OPTION = OPT = AllAssetType.OPTION
    FOREX = FX = AllAssetType.FOREX
    CRYPTOCURRENCY = CRYPTO = AllAssetType.CRYPTO
    COMMODITY = CMDTY = AllAssetType.COMMODITY
    ETF = AllAssetType.ETF
    FUND = AllAssetType.FUND  # mutual fund
    BOND = AllAssetType.BOND
    INDEX = AllAssetType.INDEX

    
class CryptoAssetType(StrEnum):
    FUTURE = FUT = AllAssetType.FUTURE
    PERPETUAL = PERP = AllAssetType.PERPETUAL
    OPTION = OPT = AllAssetType.OPTION
    CRYPTOCURRENCY = CRYPTO = SPOT = AllAssetType.CRYPTO
    INDEX = AllAssetType.INDEX


# TODO: add DeFi asset types
class DappAssetType(StrEnum):
    INDEX = AllAssetType.INDEX


# EXTEND
ASSET_TYPE_ALIASES: dict[str, str] = {
    "IPERP": f'{AssetTypeModifier.INVERSE}-{AllAssetType.PERPETUAL}',
    "IPERPETUAL": f'{AssetTypeModifier.INVERSE}-{AllAssetType.PERPETUAL}',
    "IFUT": f'{AssetTypeModifier.INVERSE}-{AllAssetType.FUTURE}',
    "IFUTURE": f'{AssetTypeModifier.INVERSE}-{AllAssetType.FUTURE}',
    # "IOPT": f'{AssetTypeModifier.INVERSE}-{AllAssetType.OPTION}',
    # "IOPTION": f'{AssetTypeModifier.INVERSE}-{AllAssetType.OPTION}',
}