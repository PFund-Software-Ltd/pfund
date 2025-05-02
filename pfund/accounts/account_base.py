from pfund.enums import Environment, Broker


class BaseAccount:
    num = 0

    @classmethod
    def add_account_num(cls):
        cls.num += 1
        return str(cls.num)
    
    def __init__(self, env: Environment, bkr: Broker, name: str=''):
        self.env = env
        self.bkr = bkr
        name = name or self._get_default_name()
        self.name = self.acc = name.upper()

    def _get_default_name(self):
        return self.__class__.__name__ + '-' + self.add_account_num()
    
    def __str__(self):
        return f'Broker={self.bkr.value}|Account={self.name}'

    def __repr__(self):
        return f'{self.bkr.value}:{self.name}'

    def __eq__(self, other):
        if not isinstance(other, BaseAccount):
            return NotImplemented  # Allow other types to define equality with BaseProduct
        return (
            self.env == other.env
            and self.bkr == other.bkr
            and self.name == other.name
        )
        
    def __hash__(self):
        return hash((self.env, self.bkr, self.name))