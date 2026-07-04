"""
NOTE:
Margin Balance and Available Balance in IB include both the cash balance in the account,
as well as any other financial instruments that can be used as collateral for margin loans,
such as stocks or bonds.
So they are not actually your amount of e.g. USD cash, but the total value of your cash + stock + other products in USD

Margin Balance = Equity with Loan Value (ELV) in IB
Available Balance = Available Funds in IB
"""

from pfund.entities import BaseBalance


class InteractiveBrokersBalance(BaseBalance):
    pass
