"""
NOTE: 
Margin Balance and Available Balance in IB include both the cash balance in the account,
as well as any other financial instruments that can be used as collateral for margin loans,
such as stocks or bonds.
So they are not actually your amount of e.g. USD cash, but the total value of your cash + stock + other products in USD

Margin Balance = Equity with Loan Value (ELV) in IB
Available Balance = Available Funds in IB
For more details, please refer to: 
1. https://ibkr.info/node/1445/
2. https://www.interactivebrokers.ca/en/index.php?f=4745&p=overview3
"""

from pfund.balances.balance_base import BaseBalance


class IBBalance(BaseBalance):
    def __init__(self, account, ccy):
        super().__init__(account, ccy)

    def __str__(self):
        return f'Broker={self.bkr}|Account={self.acc}|Currency={self.ccy}|Balance={self._balance}'
    
    def __repr__(self):
        return f'{self.bkr}:{self.acc}:{self.ccy}:{self._balance}'
