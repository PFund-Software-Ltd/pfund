from enum import StrEnum


class CryptoMonthCode(StrEnum):
    CW = 'CW'  # current week
    NW = 'NW'  # next week
    CM = 'CM'  # current month
    NM = 'NM'  # next month
    CQ = 'CQ'  # current quarter
    NQ = 'NQ'  # next quarter
    
    
class FutureMonthCode(StrEnum):
    F = 'F'  # January
    G = 'G'  # February
    H = 'H'  # March
    J = 'J'  # April
    K = 'K'  # May
    M = 'M'  # June
    N = 'N'  # July
    Q = 'Q'  # August
    U = 'U'  # September
    V = 'V'  # October
    X = 'X'  # November
    Z = 'Z'  # December
