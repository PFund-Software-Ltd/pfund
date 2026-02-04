from pfund.balances.balance_base import BaseBalance


class CryptoBalance(BaseBalance):
    def __init__(self, account, ccy):
        super().__init__(account, ccy)
        self.exch = account.exch

    def __str__(self):
        return f'Broker={self.bkr}|Exchange={self.exch}|Account={self.acc}|Currency={self.ccy}|Balance={self._balance}'
    
    def __repr__(self):
        return f'{self.bkr}:{self.exch}:{self.acc}:{self.ccy}:{self._balance}'
