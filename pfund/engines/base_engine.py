from __future__ import annotations
from typing import TYPE_CHECKING, Literal, overload, ClassVar
if TYPE_CHECKING:
    from mtflow.stores.mtstore import MTStore
    from pfeed.typing import tDATA_TOOL
    from pfund.typing import TradeEngineSettingsDict, BacktestEngineSettingsDict
    from pfund.typing import StrategyT, tENVIRONMENT, tBROKER, DataRangeDict
    from pfund.data_tools.data_tool_base import BaseDataTool
    from pfund.brokers.broker_base import BaseBroker
    from pfund.brokers.broker_crypto import CryptoBroker
    from pfund.brokers.ib.broker_ib import IBBroker
    from pfund.strategies.strategy_base import BaseStrategy

import logging
import datetime
import importlib

from pfund import cprint
from pfund.config import get_config
from pfund.utils.utils import Singleton
from pfund.enums import Environment, Broker


ENV_COLORS = {
    # 'yellow': 'bold yellow on #ffffe0',
    # 'magenta': 'bold magenta on #fff0ff',
    # 'TRAIN': 'bold cyan on #d0ffff',
    'BACKTEST': 'bold blue on #e0e0ff',
    'SANDBOX': 'bold black on #f0f0f0',
    'PAPER': 'bold red on #ffe0e0',
    'LIVE': 'bold green on #e0ffe0',
}

config = get_config()


class BaseEngine(Singleton):
    settings: ClassVar[TradeEngineSettingsDict | BacktestEngineSettingsDict]
        
    DataTool: type[BaseDataTool] | None = None
    _PROCESS_NO_PONG_TOLERANCE_IN_SECONDS = 30

    def __init__(
        self, 
        *,
        env: tENVIRONMENT, 
        data_tool: tDATA_TOOL='polars', 
        data_range: str | DataRangeDict='ytd', 
        use_ray: bool=False,
        settings: TradeEngineSettingsDict | BacktestEngineSettingsDict | None=None,
    ):
        from pfeed.enums import DataTool
        from pfund.managers.strategy_manager import StrategyManager

        self._initialized = True
        self._env = Environment[env.upper()]
        # FIXME:
        if env == Environment.BACKTEST:
            assert self.__class__.__name__ == 'BacktestEngine', f'{env=} is only allowed to be created using BacktestEngine'
        cprint(f"{env.value} Engine is running", style=ENV_COLORS[env.value])
        
        # TODO, do NOT allow PAPER, LIVE env for now
        assert self.env not in [Environment.PAPER, Environment.LIVE], f"env={self.env.value} is not allowed for now"
        
        self._setup_logging()
        self.logger = logging.getLogger('pfund')
        self._dataset_start, self._dataset_end = self._parse_data_range(data_range)
        self._use_ray = use_ray
        self._data_tool = DataTool[data_tool.lower()]
        
        self._store = self._create_mtstore()
        self._brokers = {}
        self.strategy_manager = StrategyManager()

        cls = self.__class__
        cls.settings.update(settings or {})
    
    @property
    def env(self) -> Environment:
        return self._env
    
    @staticmethod
    def _parse_data_range(data_range: str | DataRangeDict) -> tuple[datetime.date, datetime.date]:
        from pfeed.utils.utils import rollback_date_range
        if isinstance(data_range, str):
            rollback_period = data_range
            assert rollback_period != 'max', '"max" is not allowed for `data_range`'
            start_date, end_date = rollback_date_range(rollback_period)
        else:
            start_date = datetime.datetime.strptime(data_range['start_date'], '%Y-%m-%d').date()
            if 'end_date' not in data_range:
                yesterday = datetime.datetime.now(tz=datetime.timezone.utc).date() - datetime.timedelta(days=1)
                end_date = yesterday
            else:
                end_date = datetime.datetime.strptime(data_range['end_date'], '%Y-%m-%d').date()
        assert start_date <= end_date, f"start_date must be before end_date: {start_date} <= {end_date}"
        return start_date, end_date
    
    @property
    def dataset_start(self) -> datetime.date:
        return self._dataset_start
    
    @property
    def dataset_end(self) -> datetime.date:
        return self._dataset_end
    
    @property
    def store(self) -> MTStore:
        return self._store
    
    @property
    def brokers(self) -> dict[str, BaseBroker]:
        return self._brokers
    
    def _create_mtstore(self) -> MTStore:
        from mtflow.stores.mtstore import MTStore
        mtstore = MTStore(env=self.env.value, data_tool=self._data_tool.value)
        mtstore._set_logger(self.logger)
        return mtstore
    
    def _setup_logging(self):
        from pfund.plogging import setup_loggers
        log_path = f'{config.log_path}/{self.env.value.lower()}'
        logging_config_file_path = config.logging_config_file_path
        logging_configurator  = setup_loggers(log_path, logging_config_file_path, user_logging_config=config.logging_config)
        config.set_logging_configurator(logging_configurator)
        
    def _init_ray(self, **kwargs):
        import ray
        if not ray.is_initialized():
            ray.init(**kwargs)

    def _shutdown_ray(self):
        import ray
        if ray.is_initialized():
            ray.shutdown()
    
    def run(self):
        self._store.freeze()
    
    def get_strategy(self, strat: str) -> BaseStrategy | None:
        return self.strategy_manager.get_strategy(strat)

    def add_strategy(self, strategy: StrategyT, resolution: str, name: str='') -> StrategyT:
        self._store.add_trading_store(strategy.name)
        return self.strategy_manager.add_strategy(strategy, resolution, name=name)

    def remove_strategy(self, strat: str):
        return self.strategy_manager.remove_strategy(strat)

    # conditional typing, returns the exact type of broker
    @overload
    def get_broker(self, bkr: Literal['CRYPTO']) -> CryptoBroker: ...
        
    # conditional typing, returns the exact type of broker
    @overload
    def get_broker(self, bkr: Literal['IB']) -> IBBroker: ...

    def get_broker(self, bkr: tBROKER) -> BaseBroker:
        return self._brokers[bkr.upper()]
    
    def add_broker(self, bkr: str) -> BaseBroker:
        bkr = bkr.upper()
        if bkr in self.brokers:
            return self.get_broker(bkr)
        broker = self._create_broker(bkr)
        self._brokers[bkr] = broker
        self.logger.debug(f'added broker {bkr}')
        return broker
    
    def remove_broker(self, bkr: tBROKER) -> BaseBroker:
        broker = self._brokers.pop(bkr.upper())
        self.logger.debug(f'removed broker {bkr}')
        return broker
    
    def get_Broker(self, bkr: tBROKER) -> type[BaseBroker]:
        broker = Broker[bkr.upper()]
        if broker == Broker.CRYPTO:
            BrokerClass = getattr(importlib.import_module('pfund.brokers.broker_crypto'), 'CryptoBroker')
        elif broker == Broker.IB:
            BrokerClass = getattr(importlib.import_module('pfund.brokers.ib.broker_ib'), 'IBBroker')
        else:
            raise ValueError(f'broker {bkr} is not supported')
        return BrokerClass