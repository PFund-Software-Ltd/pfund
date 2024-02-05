import logging
from collections import defaultdict

from pfund.const.commons import SUPPORTED_ENVIRONMENTS
from pfund.managers import *
from pfund.utils.utils import get_engine_class


class BaseBroker:
    def __init__(self, env, name):
        self.env = env.upper()
        self.logger = logging.getLogger('pfund')
        assert self.env in SUPPORTED_ENVIRONMENTS, f'env={self.env} is not supported'
        Engine = get_engine_class()
        self._settings = Engine.settings
        self.name = self.bkr = name.upper()
        self.products = defaultdict(dict)
        self.accounts = defaultdict(dict)
