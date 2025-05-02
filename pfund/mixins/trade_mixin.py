from __future__ import annotations
from typing import TYPE_CHECKING, overload, Literal
if TYPE_CHECKING:
    import torch
    import numpy as np
    import pandas as pd
    import polars as pl
    from mtflow.stores.trading_store import TradingStore
    from pfeed.typing import tDATA_SOURCE
    from pfund.typing import ModelT, IndicatorT, FeatureT, DataConfigDict
    from pfund.engines.base_engine import BaseEngine
    from pfund.typing import tTRADING_VENUE, tBROKER, tCRYPTO_EXCHANGE
    from pfund.datas.data_base import BaseData
    from pfund.datas.data_time_based import TimeBasedData
    from pfund.brokers.broker_trade import BaseBroker
    from pfund.brokers.broker_crypto import CryptoBroker
    from pfund.brokers.ib.broker_ib import IBBroker
    from pfund.products.product_base import BaseProduct
    from pfund.products.product_ib import IBProduct
    from pfund.data_tools.data_tool_base import BaseDataTool
    from pfund.datas import QuoteData, TickData, BarData
    from pfund.strategies.strategy_base import BaseStrategy
    from pfund.models.model_base import BaseModel, BaseFeature
    from pfund.indicators.indicator_base import BaseIndicator

import sys
import time
import datetime
import importlib
from pathlib import Path

from pfund.enums import CryptoExchange, Broker, ComponentType
from pfund.datas.resolution import Resolution
from pfund.utils.utils import load_yaml_file, convert_ts_to_dt
from pfund.data_tools.data_config import DataConfig


class TradeMixin:
    _file_path: Path | None = None  # Get the file path where the strategy was defined
    config = {}

    @classmethod
    def load_config(cls, config: dict | None=None):
        if config:
            cls.config = config
        elif cls._file_path:
            for file_name in ['config.yml', 'config.yaml']:
                if config := load_yaml_file(cls._file_path.parent / file_name):
                    cls.config = config
                    break
    
    def load_params(self: BaseStrategy | BaseModel, params: dict | None=None):
        if params:
            self.params = params
        elif self._file_path:
            for file_name in ['params.yml', 'params.yaml']:
                if params := load_yaml_file(self._file_path.parent / file_name):
                    self.params = params
                    break
    
    def __new__(cls, *args, **kwargs):
        if not cls._file_path:
            module = sys.modules[cls.__module__]
            if file_path := getattr(module, '__file__', None):
                cls._file_path = Path(file_path)
                cls.load_config()
        return super().__new__(cls)
    
    @property
    def data_tool(self: BaseStrategy | BaseModel) -> BaseDataTool:
        return self._data_tool
    dtl = data_tool
    
    def _create_data_tool(self: BaseStrategy | BaseModel) -> BaseDataTool:
        from pfund.engines.trade_engine import TradeEngine
        
        data_tool = TradeEngine.data_tool
        DataTool: type[BaseDataTool] = getattr(importlib.import_module(f'pfund.data_tools.data_tool_{data_tool}'), f'{data_tool.value.capitalize()}DataTool')
        return DataTool()
    
    @property
    def store(self: BaseStrategy | BaseModel) -> TradingStore:
        return self._store
    
    def get_df(
        self: BaseStrategy | BaseModel, 
        start_idx: int=0, 
        end_idx: int | None=None, 
        product: str | None=None, 
        resolution: str | None=None, 
        copy: bool=True
    ) -> pd.DataFrame | pl.LazyFrame | None:
        return self.data_tool.get_df(
            start_idx=start_idx, 
            end_idx=end_idx, 
            product=product,
            resolution=resolution,
            copy=copy
        )
    
    def _set_resolution(self, resolution: str):
        if not self._resolution:
            self._resolution = Resolution(resolution)
        else:
            raise ValueError(f"{self.name} already has a resolution {self._resolution}, cannot set to {resolution}")
        
    @property
    def component_type(self) -> ComponentType:
        from pfund.strategies.strategy_base import BaseStrategy
        from pfund.models.model_base import BaseModel, BaseFeature
        from pfund.indicators.indicator_base import BaseIndicator
        if isinstance(self, BaseStrategy):
            return ComponentType.strategy
        elif isinstance(self, BaseIndicator):
            return ComponentType.indicator
        elif isinstance(self, BaseFeature):
            return ComponentType.feature
        elif isinstance(self, BaseModel):
            return ComponentType.model
        
    @property
    def resolution(self) -> Resolution | None:
        return self._resolution
    
    @property
    def datas(self: BaseStrategy | BaseModel):
        return self._datas
    
    @property
    def dataset_start(self: BaseStrategy | BaseModel) -> datetime.date:
        return self._engine.dataset_start
    
    @property
    def dataset_end(self: BaseStrategy | BaseModel) -> datetime.date:
        return self._engine.dataset_end
        
    # NOTE: df = X + predictions generated by other strategies/models
    @property
    def df(self: BaseStrategy | BaseModel):
        # TODO: self.store.load_data(...)
        return self.get_df(copy=False)

    @property
    def signals(self: BaseStrategy | BaseModel):
        return self._signals
    
    @property
    def INDEX(self: BaseStrategy | BaseModel):
        return self.data_tool.INDEX
    
    @property
    def GROUP(self: BaseStrategy | BaseModel):
        return self.data_tool.GROUP
    
    @property
    def name(self: BaseStrategy | BaseModel):
        return self._name
    
    @property
    def components(self: BaseStrategy | BaseModel) -> list[BaseStrategy | BaseModel | BaseFeature | BaseIndicator]:
        components = [*self.models.values(), *self.features.values(), *self.indicators.values()]
        if self.is_strategy():
            components.extend([*self.strategies.values()])
        return components

    @staticmethod
    def dt(ts: float) -> datetime.datetime:
        return convert_ts_to_dt(ts)
    
    @staticmethod
    def now() -> datetime.datetime:
        return datetime.datetime.now(tz=datetime.timezone.utc)
    
    @staticmethod
    def get_delay(ts: float) -> float:
        return time.time() - ts
    
    @staticmethod
    def is_ray_actor(value) -> bool:
        from ray.actor import ActorClass
        return isinstance(value, ActorClass)
    
    def is_strategy(self: BaseStrategy | BaseModel) -> bool:
        return self.component_type == ComponentType.strategy
    
    def is_model(self: BaseStrategy | BaseModel) -> bool:
        return self.component_type == ComponentType.model
    
    def is_indicator(self: BaseStrategy | BaseModel) -> bool:
        return self.component_type == ComponentType.indicator
    
    def is_feature(self: BaseStrategy | BaseModel) -> bool:
        return self.component_type == ComponentType.feature
    
    def is_running(self: BaseStrategy | BaseModel):
        return self._is_running
    
    def _set_engine(self: BaseStrategy | BaseModel, engine: BaseEngine | None):
        self._engine: BaseEngine | None = engine
    
    def _create_logger(self: BaseStrategy | BaseModel):
        from pfund._logging import create_dynamic_logger
        self.logger = create_dynamic_logger(self.name, self.component_type)
        self.store.set_logger(self.logger)
        
    def _setup_zmq(self: BaseStrategy | BaseModel):
        import zmq
        from mtflow.messaging.zeromq import ZeroMQ

        zmq_urls = self._engine.settings.zmq_urls
        self._zmq = ZeroMQ(
            url=zmq_urls.get(self.name, ZeroMQ.DEFAULT_URL),
            receiver_socket_type=zmq.SUB,  # receive data from engine
            sender_socket_type=zmq.PUSH,  # send e.g. orders to engine
        )
        # TODO: subscribe to selected topics, e.g. b'BYBIT:orderbook:BTCUSDT'
        self._zmq.setsockopt(zmq.SUBSCRIBE, b'')
        
    def _prepare_df(self: BaseStrategy | BaseModel):
        return self.data_tool.prepare_df(ts_col_type='timestamp')
    
    def _append_to_df(self: BaseStrategy | BaseModel, data: BaseData, **extra_data):
        return self.data_tool.append_to_df(data, self.predictions, **extra_data)
    
    def _get_default_name(self: BaseStrategy | BaseModel):
        return self.__class__.__name__
    
    def _set_name(self, name: str):
        self._name = name.lower()
        if self.component_type.lower() not in self._name:
            self._name += f"_{self.component_type}"
    
    def get_default_signal_cols(self: BaseStrategy | BaseModel, num_cols: int) -> list[str]:
        if num_cols == 1:
            columns = [self.name]
        else:
            columns = [f'{self.name}-{i}' for i in range(num_cols)]
        return columns
   
    def get_signal_cols(self: BaseStrategy | BaseModel) -> list[str]:
        return self._signal_cols
    
    def set_signal_cols(self: BaseStrategy | BaseModel, columns: list[str]):
        self._signal_cols = [f'{self.name}-{col}' if not col.startswith(self.name) else col for col in columns]
        self._num_signal_cols = len(columns)
        
    def _set_consumer(self: BaseStrategy | BaseModel, consumer: BaseStrategy | BaseModel):
        if not self._consumer:
            self._consumer = consumer
        else:
            raise ValueError(f"{self.name} already has a consumer {self._consumer.name}")
            
    def _add_listener(self: BaseStrategy | BaseModel, listener: BaseStrategy | BaseModel, data: BaseData):
        if listener not in self._listeners[data]:
            self._listeners[data].append(listener)
    
    def _derive_bkr_from_trading_venue(self: BaseStrategy | BaseModel, trading_venue: tTRADING_VENUE) -> tBROKER:
        trading_venue = trading_venue.upper()
        return 'CRYPTO' if trading_venue in CryptoExchange.__members__ else trading_venue
    
    @overload
    def get_broker(self: BaseStrategy | BaseModel, bkr: Literal['CRYPTO']) -> CryptoBroker: ...
        
    @overload
    def get_broker(self: BaseStrategy | BaseModel, bkr: Literal['IB']) -> IBBroker: ...
    
    def get_broker(self: BaseStrategy | BaseModel, trading_venue_or_broker: tBROKER | tTRADING_VENUE) -> BaseBroker:
        if trading_venue_or_broker in Broker.__members__:
            bkr = trading_venue_or_broker
        else:
            bkr = self._derive_bkr_from_trading_venue(trading_venue_or_broker)
        return self._engine.get_broker(bkr)
    
    def list_brokers(self: BaseStrategy | BaseModel) -> list[BaseBroker]:
        return list(self._engine.brokers.values())
    
    @overload
    def get_product(self: BaseStrategy | BaseModel, trading_venue: tCRYPTO_EXCHANGE, pdt: str, exch: str='') -> BaseProduct | None: ...
        
    @overload
    def get_product(self: BaseStrategy | BaseModel, trading_venue: Literal['IB'], pdt: str, exch: str='') -> IBProduct | None: ...
    
    def get_product(self: BaseStrategy | BaseModel, trading_venue: tTRADING_VENUE, pdt: str, exch: str='') -> BaseProduct | None:
        broker = self.get_broker(trading_venue)
        if broker.name == 'CRYPTO':
            exch = trading_venue
            product: BaseProduct | None = broker.get_product(exch, pdt)
        else:
            product: BaseProduct | None = broker.get_product(pdt, exch=exch)
        if product and product not in self._datas:
            self.logger.warning(f"{self.name} is getting '{product}' that is not in its datas")
        return product
    
    def list_products(self: BaseStrategy | BaseModel) -> list[BaseProduct]:
        return list(self._datas.keys())

    def _parse_data_config(self: BaseStrategy | BaseModel, data_config: DataConfigDict | DataConfig | None) -> DataConfig:
        if isinstance(data_config, DataConfig):
            return data_config
        data_config = data_config or {}
        data_config['primary_resolution'] = self.resolution
        data_config = DataConfig(**data_config)
        return data_config
    
    # TODO
    def add_custom_data(self: BaseStrategy | BaseModel):
        raise NotImplementedError

    def list_datas(self: BaseStrategy | BaseModel) -> list[TimeBasedData]:
        datas = []
        for product in self._datas:
            datas.extend(list(self._datas[product].values()))
        return datas
        
    def _add_data(self: BaseStrategy | BaseModel, data: TimeBasedData):
        if data.is_quote():
            self._orderbooks[data.product] = data
        elif data.is_tick():
            self._tradebooks[data.product] = data
        self._datas[data.product][data.resol] = data
    
    def _add_data_to_consumer(
        self: BaseStrategy | BaseModel, 
        trading_venue: tTRADING_VENUE, 
        product: str, 
        symbol: str='',
        data_source: tDATA_SOURCE | None=None,
        data_origin: str='', 
        data_config: DataConfigDict | DataConfig | None=None,
        **product_specs
    ) -> list[TimeBasedData]:
        datas: list[TimeBasedData] = self._consumer.add_data(
            trading_venue=trading_venue,
            product=product, 
            symbol=symbol,
            data_source=data_source, 
            data_origin=data_origin, 
            data_config=data_config, 
            **product_specs
        )
        for data in datas:
            self._add_data(data)
            self._consumer._add_listener(self, data)
        return datas
    
    def _add_datas_from_consumer_if_none(self: BaseStrategy | BaseModel) -> list[BaseData]:
        has_no_data = self._consumer and not self._datas
        if not has_no_data:
            return []
        self.logger.info(f"No data for {self.name}, adding datas from consumer {self._consumer.name}")
        datas: list[TimeBasedData] = self._consumer.list_datas()
        for data in datas:
            self._add_data(data)
            self._consumer._add_listener(self, data)
        return datas

    def get_data(self: BaseStrategy | BaseModel, product: BaseProduct, resolution: str | Resolution) -> BaseData | None:
        if isinstance(resolution, str):
            resolution = Resolution(resolution)
        resolution_repr = repr(resolution)
        return self._datas[product].get(resolution_repr, None)
    
    def get_orderbook(self: BaseStrategy | BaseModel, product: BaseProduct) -> BaseData:
        return self._orderbooks[product]
    
    def get_tradebook(self: BaseStrategy | BaseModel, product: BaseProduct) -> BaseData:
        return self._tradebooks[product]
    
    def get_model(self: BaseStrategy | BaseModel, name: str) -> BaseModel:
        return self.models[name]
    
    def _add_model_component(
        self: BaseStrategy | BaseModel, 
        component: ModelT | FeatureT | IndicatorT,
        name: str='',
        min_data: None | int=None,
        max_data: None | int=None,
        group_data: bool=True,
        signal_cols: list[str] | None=None,
    ) -> ModelT | FeatureT | IndicatorT:
        '''Adds a model to the current model.
        Args:
            min_data (int): Minimum number of data points required for the model to make a prediction.
            max_data (int | None): Maximum number of data points required for the model to make a prediction.
            - If None: max_data is set to min_data.
            - If value=-1: include all data
            
            group_data (bool): Determines how `min_data` and `max_data` are applied to the whole df:
            - If True: `min_data` and `max_data` apply to each group=(product, resolution).
            e.g. if `min_data=2`, at least two data points are required for each group=(product, resolution).
            - If False: `min_data` and `max_data` apply to the entire dataset, not segregated by product or resolution.
            e.g. if `min_data=2`, at least two data points are required for the whole dataset.
            
            signal_cols: signal columns, if not provided, it will be derived in predict()
        '''
        Model = component.get_ml_model_type()
        assert isinstance(component, Model), \
            f"{component.component_type} '{component.__class__.__name__}' is not an instance of {Model.__name__}. Please create your {component.component_type} using 'class {component.__class__.__name__}({Model.__name__})'"
        if name:
            component._set_name(name)
        component._create_logger()
        component._set_consumer(self)
        component._set_resolution(self._consumer.resolution)
        if min_data:
            component.set_min_data(min_data)
        if max_data:
            component.set_max_data(max_data)
        component.set_group_data(group_data)
        if signal_cols:
            component.set_signal_cols(signal_cols)
        
        if component.is_model():
            components = self.models
        elif component.is_feature():
            components = self.features
        elif component.is_indicator():
            components = self.indicators

        if component.name in components:
            raise ValueError(f"{component.name} already exists")
        else:
            components[component.name] = component
            self.logger.debug(f"added {component.name}")
            return component
    
    def _remove_model_component(self: BaseStrategy | BaseModel, name: str, component_type: ComponentType):
        # get e.g. self.models, self.features, self.indicators
        components = getattr(self, component_type.value+'s')
        if name in components:
            del components[name]
            self.logger.debug(f"removed '{name}'")
        else:
            self.logger.error(f"'{name}' cannot be found, failed to remove")
    
    def add_model(
        self: BaseStrategy | BaseModel, 
        model: ModelT, 
        name: str='',
        min_data: None | int=None,
        max_data: None | int=None,
        group_data: bool=True,
        signal_cols: list[str] | None=None,
    ) -> ModelT:
        return self._add_model_component(model, name, min_data, max_data, group_data, signal_cols)
    
    def remove_model(self: BaseStrategy | BaseModel, name: str):
        self._remove_model_component(name, ComponentType.model)
    
    def add_feature(
        self: BaseStrategy | BaseModel, 
        feature: FeatureT, 
        name: str='',
        min_data: None | int=None,
        max_data: None | int=None,
        group_data: bool=True,
        signal_cols: list[str] | None=None,
    ) -> FeatureT:
        return self._add_model_component(
            feature, 
            name=name, 
            min_data=min_data, 
            max_data=max_data, 
            group_data=group_data,
            signal_cols=signal_cols,
        )
    
    def remove_feature(self: BaseStrategy | BaseModel, name: str):
        self._remove_model_component(name, ComponentType.feature)
    
    def add_indicator(
        self: BaseStrategy | BaseModel, 
        indicator: IndicatorT, 
        name: str='',
        min_data: None | int=None,
        max_data: None | int=None,
        group_data: bool=True,
        signal_cols: list[str] | None=None,
    ) -> IndicatorT:
        return self._add_model_component(
            indicator, 
            name=name, 
            min_data=min_data, 
            max_data=max_data, 
            group_data=group_data,
            signal_cols=signal_cols,
        )

    def remove_indicator(self: BaseStrategy | BaseModel, name: str):
        self._remove_model_component(name, ComponentType.indicator)
        
    def update_quote(self: BaseStrategy | BaseModel, data: QuoteData, **extra_data):
        product, bids, asks, ts = data.product, data.bids, data.asks, data.ts
        for listener in self._listeners[data]:
            listener.update_quote(data, **extra_data)
            self.update_predictions(data, listener)
        self._append_to_df(data, **extra_data)
        self.on_quote(product, bids, asks, ts, **extra_data)

    def update_tick(self: BaseStrategy | BaseModel, data: TickData, **extra_data):
        product, px, qty, ts = data.product, data.px, data.qty, data.ts
        for listener in self._listeners[data]:
            listener.update_tick(data, **extra_data)
            self.update_predictions(data, listener)
        self._append_to_df(data, **extra_data)
        self.on_tick(product, px, qty, ts, **extra_data)
    
    def update_bar(self: BaseStrategy | BaseModel, data: BarData, **extra_data):
        product, bar, ts = data.product, data.bar, data.bar.end_ts
        for listener in self._listeners[data]:
            # NOTE: listener could be a strategy or a model
            listener.update_bar(data, **extra_data)
            self.update_predictions(data, listener)
        self._append_to_df(data, **extra_data)
        self.on_bar(product, bar, ts, **extra_data)

    def update_predictions(self: BaseStrategy | BaseModel, data: BaseData, listener: BaseStrategy | BaseModel):
        pred_y: torch.Tensor | np.ndarray | None = listener._next(data)
        if pred_y is not None:
            signal_cols = listener.get_signal_cols()
            for i, col in enumerate(signal_cols):
                self.predictions[col] = pred_y[i]
                
    def _start_models(self: BaseStrategy | BaseModel):
        for model in self.models.values():
            model.start()
    
    
    '''
    ************************************************
    Custom Functions
    Users can customize these functions in their strategies/models.
    ************************************************
    '''
    def add_datas(self: BaseStrategy | BaseModel):
        pass
    
    def add_models(self: BaseStrategy | BaseModel):
        pass

    def add_features(self: BaseStrategy | BaseModel):
        pass
    
    def add_indicators(self: BaseStrategy | BaseModel):
        pass
    
    def on_start(self: BaseStrategy | BaseModel):
        pass
    
    def on_stop(self: BaseStrategy | BaseModel):
        pass
    
    
    '''
    ************************************************
    Sugar Functions
    ************************************************
    '''
    def get_second_bar(self: BaseStrategy | BaseModel, product: BaseProduct, period: int) -> BarData | None:
        return self.get_data(product, resolution=f'{period}s')
    
    def get_minute_bar(self: BaseStrategy | BaseModel, product: BaseProduct, period: int) -> BarData | None:
        return self.get_data(product, resolution=f'{period}m')
    
    def get_hour_bar(self: BaseStrategy | BaseModel, product: BaseProduct, period: int) -> BarData | None:
        return self.get_data(product, resolution=f'{period}h')
    
    def get_day_bar(self: BaseStrategy | BaseModel, product: BaseProduct, period: int) -> BarData | None:
        return self.get_data(product, resolution=f'{period}d')
    
    def get_week_bar(self: BaseStrategy | BaseModel, product: BaseProduct, period: int) -> BarData | None:
        return self.get_data(product, resolution=f'{period}w')
    
    def get_month_bar(self: BaseStrategy | BaseModel, product: BaseProduct, period: int) -> BarData | None:
        return self.get_data(product, resolution=f'{period}M')