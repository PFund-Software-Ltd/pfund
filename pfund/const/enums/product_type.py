from enum import StrEnum


class ProductType(StrEnum):
    STK = 'STK'
    FUT = 'FUT'
    ETF = 'ETF'
    OPT = 'OPT'
    FX = 'FX'
    CRYPTO = 'CRYPTO'
    BOND = 'BOND'
    MTF = 'MTF'  # mutual fund
    CMDTY = 'CMDTY'  # commodity
    PERP = 'PERP'
    IPERP = 'IPERP'  # inverse perpetual
    SPOT = 'SPOT'
    IFUT = 'IFUT'  # inverse future
    INDEX = 'INDEX'

    
class TradFiProductType(StrEnum):
    STK = ProductType.STK
    FUT = ProductType.FUT
    ETF = ProductType.ETF
    OPT = ProductType.OPT
    FX = ProductType.FX
    CRYPTO = ProductType.CRYPTO
    BOND = ProductType.BOND
    MTF = ProductType.MTF
    CMDTY = ProductType.CMDTY
    INDEX = ProductType.INDEX
    
    
class CeFiProductType(StrEnum):
    PERP = ProductType.PERP
    IPERP = ProductType.IPERP
    SPOT = ProductType.SPOT
    FUT = ProductType.FUT
    IFUT = ProductType.IFUT
    OPT = ProductType.OPT
    INDEX = ProductType.INDEX


# TODO: add DeFi product types
class DeFiProductType(StrEnum):
    INDEX = ProductType.INDEX
