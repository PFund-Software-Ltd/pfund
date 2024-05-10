from __future__ import annotations

import os
import logging
import importlib

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund.types.core import tStrategy
    from pfund.types.common_literals import tSUPPORTED_DATA_TOOLS
    
from rich.console import Console

from pfund.utils.utils import Singleton
from pfund.strategies.strategy_base import BaseStrategy
from pfund.brokers.broker_base import BaseBroker
from pfund.managers.strategy_manager import StrategyManager
from pfund.const.common import SUPPORTED_ENVIRONMENTS, SUPPORTED_BROKERS, SUPPORTED_DATA_TOOLS
from pfund.config_handler import ConfigHandler
from pfund.plogging import set_up_loggers
from pfund.plogging.config import LoggingDictConfigurator


ENV_COLORS = {
    # 'yellow': 'bold yellow on #ffffe0',
    # 'magenta': 'bold magenta on #fff0ff',
    'BACKTEST': 'bold blue on #e0e0ff',
    'TRAIN': 'bold cyan on #d0ffff',
    'SANDBOX': 'bold black on #f0f0f0',
    'PAPER': 'bold red on #ffe0e0',
    'LIVE': 'bold green on #e0ffe0',
}


class BaseEngine(Singleton):
    _PROCESS_NO_PONG_TOLERANCE_IN_SECONDS = 30

    def __new__(cls, env, data_tool: tSUPPORTED_DATA_TOOLS='pandas', config: ConfigHandler | None=None, **settings):
        if not hasattr(cls, 'env'):
            cls.env = env.upper() if isinstance(env, str) else str(env).upper()
            assert cls.env in SUPPORTED_ENVIRONMENTS, f'env={cls.env} is not supported'

            # TODO, do NOT allow LIVE env for now
            assert cls.env != 'LIVE', f"{cls.env} is not allowed for now, please use env='PAPER' instead"
            
            os.environ['env'] = cls.env
            Console().print(f"{cls.env} Engine is running", style=ENV_COLORS[cls.env])
        if not hasattr(cls, 'data_tool'):
            assert data_tool in SUPPORTED_DATA_TOOLS, f'{data_tool=} is not supported, {SUPPORTED_DATA_TOOLS=}'
            cls.data_tool = data_tool
        if not hasattr(cls, 'settings'):
            cls.settings = settings
        if not hasattr(cls, 'config'):
            cls.config: ConfigHandler = config if config else ConfigHandler.load_config()
            log_path = f'{cls.config.log_path}/{cls.env}'
            logging_config_file_path = cls.config.logging_config_file_path
            cls.logging_configurator: LoggingDictConfigurator  = set_up_loggers(log_path, logging_config_file_path, user_logging_config=cls.config.logging_config)
        return super().__new__(cls)
    
    def __init__(self, env, data_tool: tSUPPORTED_DATA_TOOLS='pandas', config: ConfigHandler | None=None, **settings):
        # avoid re-initialization to implement singleton class correctly
        if not hasattr(self, '_initialized'):
            self.logger = logging.getLogger('pfund')
            self.brokers = {}
            self.strategy_manager = self.sm = StrategyManager()
            self._initialized = True

    def get_strategy(self, strat: str) -> BaseStrategy | None:
        return self.strategy_manager.get_strategy(strat)

    def add_strategy(self, strategy: tStrategy, name: str='', is_parallel=False) -> tStrategy:
        return self.strategy_manager.add_strategy(strategy, name=name, is_parallel=is_parallel)

    def remove_strategy(self, strat: str):
        return self.strategy_manager.remove_strategy(strat)

    def get_broker(self, bkr: str) -> BaseBroker:
        return self.brokers[bkr.upper()]
    
    def get_Broker(self, bkr: str) -> type[BaseBroker]:
        bkr = bkr.upper()
        assert bkr in SUPPORTED_BROKERS, f'broker {bkr} is not supported'
        if bkr == 'CRYPTO':
            Broker = getattr(importlib.import_module(f'pfund.brokers.broker_{bkr.lower()}'), 'CryptoBroker')
        elif bkr == 'IB':
            Broker = getattr(importlib.import_module(f'pfund.brokers.ib.broker_{bkr.lower()}'), 'IBBroker')
        return Broker