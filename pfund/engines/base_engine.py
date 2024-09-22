from __future__ import annotations

import os
import logging
import importlib

from typing import TYPE_CHECKING, Literal, overload
if TYPE_CHECKING:
    from pfund.types.core import tStrategy
    from pfund.types.common_literals import tSUPPORTED_DATA_TOOLS, tSUPPORTED_ENVIRONMENTS, tSUPPORTED_BROKERS
    from pfund.brokers import BaseBroker, CryptoBroker, IBBroker
    from pfund.strategies.strategy_base import BaseStrategy

from rich.console import Console

from pfund.utils.utils import Singleton
from pfund.const.common import SUPPORTED_ENVIRONMENTS, SUPPORTED_BROKERS, SUPPORTED_DATA_TOOLS
from pfund.config_handler import ConfigHandler


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

    def __new__(
        cls, 
        env: tSUPPORTED_ENVIRONMENTS,
        data_tool: tSUPPORTED_DATA_TOOLS='pandas', 
        config: ConfigHandler | None=None,
        **settings
    ):
        from pfund.plogging.config import LoggingDictConfigurator
        from pfund.plogging import set_up_loggers

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
            cls.DataTool = getattr(importlib.import_module(f'pfund.data_tools.data_tool_{data_tool.lower()}'), f'{data_tool.capitalize()}DataTool')
        if not hasattr(cls, 'config'):
            cls.config: ConfigHandler = config if config else ConfigHandler.load_config()
            log_path = f'{cls.config.log_path}/{cls.env}'
            logging_config_file_path = cls.config.logging_config_file_path
            cls.logging_configurator: LoggingDictConfigurator  = set_up_loggers(log_path, logging_config_file_path, user_logging_config=cls.config.logging_config)
        if not hasattr(cls, 'settings'):
            from IPython import get_ipython
            cls.settings = settings
            if 'ipython' not in settings:
                settings['ipython'] = bool(get_ipython() is not None)
        return super().__new__(cls)
    
    def __init__(
        self,
        env: tSUPPORTED_ENVIRONMENTS,
        data_tool: tSUPPORTED_DATA_TOOLS='pandas',
        config: ConfigHandler | None=None, 
        **settings
    ):
        from pfund.managers.strategy_manager import StrategyManager
        # avoid re-initialization to implement singleton class correctly
        if not hasattr(self, '_initialized'):
            self._validate_env()
            self.logger = logging.getLogger('pfund')
            self.brokers = {}
            self.strategy_manager = self.sm = StrategyManager()
            self._initialized = True
    
    def _validate_env(self):
        env_to_engine = {
            'BACKTEST': 'BacktestEngine',
            'SANDBOX': 'SandboxEngine',
            'PAPER': 'TradeEngine',
            'LIVE': 'TradeEngine',
            'TRAIN': 'TrainEngine',
        }
        engine_name = env_to_engine[self.env]
        if self.__class__.__name__ != engine_name:
            raise ValueError(
                f"Invalid environment '{self.env}' for {self.__class__.__name__}. "
                f"Use the '{engine_name}' for the '{self.env}' environment."
            )

    def get_strategy(self, strat: str) -> BaseStrategy | None:
        return self.strategy_manager.get_strategy(strat)

    def add_strategy(self, strategy: tStrategy, name: str='', is_parallel=False) -> tStrategy:
        return self.strategy_manager.add_strategy(strategy, name=name, is_parallel=is_parallel)

    def remove_strategy(self, strat: str):
        return self.strategy_manager.remove_strategy(strat)

    # conditional typing, returns the exact type of broker
    @overload
    def get_broker(self, bkr: Literal['CRYPTO']) -> CryptoBroker: ...
        
    # conditional typing, returns the exact type of broker
    @overload
    def get_broker(self, bkr: Literal['IB']) -> IBBroker: ...

    def get_broker(self, bkr: tSUPPORTED_BROKERS) -> BaseBroker:
        return self.brokers[bkr.upper()]
    
    def remove_broker(self, bkr: tSUPPORTED_BROKERS) -> BaseBroker:
        broker = self.brokers.pop(bkr.upper())
        self.logger.debug(f'removed broker {bkr}')
        return broker
    
    def get_Broker(self, bkr: tSUPPORTED_BROKERS) -> type[BaseBroker]:
        bkr = bkr.upper()
        assert bkr in SUPPORTED_BROKERS, f'broker {bkr} is not supported'
        if bkr == 'CRYPTO':
            Broker = getattr(importlib.import_module(f'pfund.brokers.broker_{bkr.lower()}'), 'CryptoBroker')
        elif bkr == 'IB':
            Broker = getattr(importlib.import_module(f'pfund.brokers.ib.broker_{bkr.lower()}'), 'IBBroker')
        return Broker