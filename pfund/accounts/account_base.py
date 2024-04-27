class BaseAccount:
    num = 0

    @classmethod
    def add_account_num(cls):
        cls.num += 1
        return str(cls.num)
    
    def __init__(self, env: str, bkr: str, acc: str='', **kwargs):
        self.env = env.upper()
        self.bkr = bkr.upper()
        acc = acc or 'ACC-' + self.add_account_num()  # may have same oid if running multiple bots; must less than 36 chars for binance
        self.name = self.acc = acc.upper()
        for k, v in kwargs.items():
            setattr(self, k, v)
        if not hasattr(self, 'strat'):
            self.strat = ''

    def __str__(self):
        return f'Broker={self.bkr}|Account={self.name}|Strategy={self.strat}'

    def __repr__(self):
        return f'{self.bkr}-{self.name}'

    def __hash__(self):
        return hash((self.env, self.bkr, self.name))