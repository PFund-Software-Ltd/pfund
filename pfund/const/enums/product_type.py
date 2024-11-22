from enum import StrEnum


class TradFiProductType(StrEnum):
    STK = 'STK'
    FUT = 'FUT'
    ETF = 'ETF'
    OPT = 'OPT'
    FX = 'FX'
    CRYPTO = 'CRYPTO'
    BOND = 'BOND'
    MTF = 'MTF'
    CMDTY = 'CMDTY'
    
    
class CeFiProductType(StrEnum):
    PERP = 'PERP'
    IPERP = 'IPERP'
    SPOT = 'SPOT'
    FUT = 'FUT'
    IFUT = 'IFUT'
    OPT = 'OPT'


# TODO: add DeFi product types
class DeFiProductType(StrEnum):
    pass



class ProductType(StrEnum):
    STK = 'STK'
    FUT = 'FUT'
    ETF = 'ETF'
    OPT = 'OPT'
    FX = 'FX'
    CRYPTO = 'CRYPTO'
    BOND = 'BOND'
    MTF = 'MTF'
    CMDTY = 'CMDTY'
    PERP = 'PERP'
    IPERP = 'IPERP'
    SPOT = 'SPOT'
    IFUT = 'IFUT'