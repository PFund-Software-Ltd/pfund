from __future__ import annotations
from typing import TYPE_CHECKING, Literal, overload, ClassVar
if TYPE_CHECKING:
    from sklearn.model_selection._split import BaseCrossValidator
    from pfeed.typing import tDATA_TOOL, tDATA_SOURCE, tDATA_LAYER, tSTORAGE
    from pfund.typing import DataRangeDict, DatasetSplitsDict
    from pfund.typing import TradeEngineSettingsDict, BacktestEngineSettingsDict
    from pfund.typing import StrategyT, tENVIRONMENT, tBROKER
    from pfund.data_tools.data_tool_base import BaseDataTool
    from pfund.brokers.broker_base import BaseBroker
    from pfund.brokers.broker_crypto import CryptoBroker
    from pfund.brokers.ib.broker_ib import IBBroker
    from pfund.strategies.strategy_base import BaseStrategy

import os
import logging
import importlib

from pfeed.feeds.market_feed import MarketFeed
from pfeed.enums import DataSource
from pfund.config import get_config
from pfund import cprint
from pfund.utils.utils import Singleton
from pfund.enums import Environment, Broker
from pfund.datas.storage_config import StorageConfig


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
        dataset_splits: int | DatasetSplitsDict | BaseCrossValidator=721, 
        use_ray: bool=False,
        settings: TradeEngineSettingsDict | BacktestEngineSettingsDict | None=None,
    ):
        from pfund.managers.strategy_manager import StrategyManager

        self._initialized = True
        
        self.env: Environment = self._set_trading_env(env)
        if self.env == Environment.BACKTEST:
            assert self.__class__.__name__ == 'BacktestEngine', f'env={self.env} is only allowed to be created using BacktestEngine'
        
        # TODO, do NOT allow PAPER, LIVE env for now
        assert self.env not in [Environment.PAPER, Environment.LIVE], f"{self.env.value} is not allowed for now"
        
        self._setup_logging()
        self.logger = logging.getLogger('pfund')
        self._storage_config: StorageConfig | None = None
        self._use_ray = use_ray

        self._data_tool: BaseDataTool = self._create_data_tool(data_tool, data_range, dataset_splits)
        self.brokers = {}
        self.strategy_manager = StrategyManager()

        cls = self.__class__
        cls.settings.update(settings or {})
    
    def configure_storage(
        self, 
        data_layer: tDATA_LAYER, 
        from_storage: tSTORAGE,
        storage_options: dict | None=None
    ):
        '''Configure global storage config so that no need to pass repeated data configs into strategy/model.add_data()
        Args:
            storage_options: configs specific to "from_storage", for MinIO, it's access_key and secret_key etc.
        '''
        self._storage_config = StorageConfig(data_layer=data_layer, from_storage=from_storage, storage_options=storage_options)
        
    def _create_data_tool(
        self, 
        data_tool: tDATA_TOOL, 
        data_range: str | DataRangeDict, 
        dataset_splits: int | DatasetSplitsDict | BaseCrossValidator
    ) -> BaseDataTool:
        from pfeed.enums import DataTool
        data_tool = DataTool[data_tool.lower()]
        DataTool: type[BaseDataTool] = getattr(importlib.import_module(f'pfund.data_tools.data_tool_{data_tool}'), f'{data_tool.capitalize()}DataTool')
        return DataTool(data_range=data_range, dataset_splits=dataset_splits)
        
    def _setup_logging(self):
        from pfund.plogging import setup_loggers
        log_path = f'{config.log_path}/{self.env.value.lower()}'
        logging_config_file_path = config.logging_config_file_path
        logging_configurator  = setup_loggers(log_path, logging_config_file_path, user_logging_config=config.logging_config)
        config.set_logging_configurator(logging_configurator)
        
    @staticmethod
    def _set_trading_env(env: tENVIRONMENT) -> Environment:
        env = Environment[env.upper()]
        os.environ['trading_env'] = env.value
        cprint(f"{env.value} Engine is running", style=ENV_COLORS[env.value])
        return env
    
    def _init_ray(self, **kwargs):
        import ray
        if not ray.is_initialized():
            ray.init(**kwargs)

    def _shutdown_ray(self):
        import ray
        if ray.is_initialized():
            ray.shutdown()
    
    def get_feed(self, data_source: tDATA_SOURCE | DataSource, **pfeed_kwargs) -> MarketFeed:
        if isinstance(data_source, str):
            data_source = DataSource[data_source.upper()]
        DataFeed = data_source.feed_class
        assert DataFeed is not None, f"Failed to import data feed for {data_source}, make sure it has been installed using `pip install pfeed[{data_source.value.lower()}]`"
        feed = DataFeed(data_tool=self._data_tool.name.value, **pfeed_kwargs)
        if not isinstance(feed, MarketFeed):
            if hasattr(feed, 'market'):
                feed = feed.market
            else:
                raise ValueError(f"Data feed {feed} is not a MarketFeed")
        return feed
    
    def get_strategy(self, strat: str) -> BaseStrategy | None:
        return self.strategy_manager.get_strategy(strat)

    def add_strategy(self, strategy: StrategyT, resolution: str, name: str='') -> StrategyT:
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
        return self.brokers[bkr.upper()]
    
    def remove_broker(self, bkr: tBROKER) -> BaseBroker:
        broker = self.brokers.pop(bkr.upper())
        self.logger.debug(f'removed broker {bkr}')
        return broker
    
    def get_Broker(self, bkr: tBROKER) -> type[BaseBroker]:
        bkr = bkr.upper()
        assert bkr in Broker.__members__, f'broker {bkr} is not supported'
        if bkr == 'CRYPTO':
            BrokerClass = getattr(importlib.import_module(f'pfund.brokers.broker_{bkr.lower()}'), 'CryptoBroker')
        elif bkr == 'IB':
            BrokerClass = getattr(importlib.import_module(f'pfund.brokers.ib.broker_{bkr.lower()}'), 'IBBroker')
        return BrokerClass