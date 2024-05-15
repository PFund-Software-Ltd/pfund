import logging
from collections import defaultdict

from pfund.const.common import SUPPORTED_ENVIRONMENTS 
from pfund.utils.utils import get_engine_class


class BaseBroker:
    def __init__(self, env, name):
        self.env = env.upper()
        self.logger = logging.getLogger('pfund')
        assert self.env in SUPPORTED_ENVIRONMENTS, f'env={self.env} is not supported'
        Engine = get_engine_class()
        self._settings = Engine.settings
        self.name = self.bkr = name.upper()
        self._products = defaultdict(dict)  # {exch: {pdt1: product1, pdt2: product2, exch1_pdt3: product, exch2_pdt3: product} }
        self._accounts = defaultdict(dict)  # {trading_venue: {acc1: account1, acc2: account2} }
    
    @property
    def products(self):
        return self._products
    
    @property
    def accounts(self):
        return self._accounts
    