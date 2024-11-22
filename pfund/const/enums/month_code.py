from enum import StrEnum


class CryptoMonthCode(StrEnum):
    CW = 'Current Week'
    NW = 'Next Week'
    CM = 'Current Month'
    NM = 'Next Month'
    CQ = 'Current Quarter'
    NQ = 'Next Quarter'
    
    
class FuturesMonthCode(StrEnum):
    F = 'January'
    G = 'February'
    H = 'March'
    J = 'April'
    K = 'May'
    M = 'June'
    N = 'July'
    Q = 'August'
    U = 'September'
    V = 'October'
    X = 'November'
    Z = 'December'
