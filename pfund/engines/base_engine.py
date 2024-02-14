import os
import logging
import importlib

from rich.console import Console

from pfund.utils.utils import Singleton
from pfund.data_tools.data_tool_base import DataTool
from pfund.strategies.strategy_base import BaseStrategy
from pfund.brokers.broker_base import BaseBroker
from pfund.managers.strategy_manager import StrategyManager
from pfund.const.commons import *
from pfund.config_handler import ConfigHandler
from pfund.plogging import set_up_loggers


ENV_COLORS = {
    'BACKTEST': 'bold blue',
    'TRAIN': 'bold cyan',
    'TEST': 'bold black',
    'PAPER': 'bold red',
    'LIVE': 'bold green',
}


class BaseEngine(Singleton):
    _PROCESS_NO_PONG_TOLERANCE_IN_SECONDS = 30

    def __new__(cls, env, data_tool: DataTool='pandas', config: ConfigHandler | None=None, **settings):
        if not hasattr(cls, 'env'):
            cls.env = env.upper() if type(env) is str else str(env).upper()
            assert cls.env in SUPPORTED_ENVIRONMENTS, f'env={cls.env} is not supported'

            # TODO, do NOT allow LIVE env for now
            assert cls.env != 'LIVE', f"{cls.env} is not allowed for now, please use env='PAPER' instead"
            
            os.environ['env'] = cls.env
            Console().print(f"{cls.env} Engine is running", style=ENV_COLORS[cls.env])
        if not hasattr(cls, 'data_tool'):
            # TODO, now supports pandas only
            assert data_tool == 'pandas', f'{data_tool=} is not supported'
            cls.data_tool = data_tool
        if not hasattr(cls, 'settings'):
            cls.settings = settings
        if not hasattr(cls, 'config'):
            cls.config = config if config else ConfigHandler.load_config()
            log_path = f'{cls.config.log_path}/{cls.env}'
            logging_config_file_path = cls.config.logging_config_file_path
            set_up_loggers(log_path, logging_config_file_path, user_logging_config=cls.config.logging_config)
        return super().__new__(cls)
    
    def __init__(self, env, data_tool: DataTool='pandas', config: ConfigHandler | None=None, **settings):
        # avoid re-initialization to implement singleton class correctly
        if not hasattr(self, '_initialized'):
            self.logger = logging.getLogger('pfund')
            DataTool = getattr(importlib.import_module(f'pfund.data_tools.data_tool_{data_tool}'), f'{data_tool.capitalize()}DataTool')
            self.data_tool = DataTool()
            self.brokers = {}
            self.strategy_manager = self.sm = StrategyManager()
            self._initialized = True

    def __getattr__(self, attr):
        '''gets triggered only when the attribute is not found'''
        try:
            return getattr(self.data_tool, attr)
        except AttributeError:
            raise AttributeError(f"'{self.__class__.__name__}' object or '{self.__class__.__name__}.data_tool' has no attribute '{attr}'")
    
    def get_strategy(self, strat: str) -> BaseStrategy | None:
        return self.strategy_manager.get_strategy(strat)

    def add_strategy(self, strategy: BaseStrategy, name: str='', is_parallel=False) -> BaseStrategy:
        return self.strategy_manager.add_strategy(strategy, name=name, is_parallel=is_parallel)

    def remove_strategy(self, strat: str):
        return self.strategy_manager.remove_strategy(strat)

    def get_broker(self, bkr: str) -> BaseBroker:
        return self.brokers[bkr.upper()]
    
    def get_Broker(self, bkr: str) -> type[BaseBroker]:
        bkr = bkr.upper()
        assert bkr in SUPPORTED_BROKERS, f'broker {bkr} is not supported'
        if bkr == 'CRYPTO':
            Broker = getattr(importlib.import_module(f'pfund.brokers.broker_{bkr.lower()}'), f'CryptoBroker')
        elif bkr == 'IB':
            Broker = getattr(importlib.import_module(f'pfund.brokers.ib.broker_{bkr.lower()}'), f'IBBroker')
        return Broker