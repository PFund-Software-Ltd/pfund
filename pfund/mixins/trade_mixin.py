from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import torch
    import numpy as np
    import pandas as pd
    import polars as pl
    from mtflow.stores.trading_store import TradingStore
    from pfeed.typing import tDATA_SOURCE
    from pfund.datas.databoy import DataBoy
    from pfund.typing import ModelT, IndicatorT, FeatureT, DataConfigDict
    from pfund.typing import tTRADING_VENUE, Component
    from pfund.datas.data_bar import Bar
    from pfund.datas.data_base import BaseData
    from pfund._logging.config import LoggingDictConfigurator
    from pfund.datas.data_time_based import TimeBasedData
    from pfund.products.product_base import BaseProduct
    from pfund.data_tools.data_tool_base import BaseDataTool
    from pfund.datas import QuoteData, TickData, BarData
    from pfund.engines.base_engine import BaseEngine
    from pfund.strategies.strategy_base import BaseStrategy
    from pfund.models.model_base import BaseModel
    from pfund.features.feature_base import BaseFeature
    from pfund.indicators.indicator_base import BaseIndicator

import time
import logging
import datetime

from pfund.datas.resolution import Resolution
from pfund.datas.data_config import DataConfig
from pfund.proxies.engine_proxy import EngineProxy
from pfund.proxies.actor_proxy import ActorProxy
from pfund.utils.utils import load_yaml_file
from pfund.enums import ComponentType, RunMode


class TradeMixin:
    config = {}
    params = {}

    # custom post init for-common attributes of strategy and model
    def __post_init__(self: Component, *args, **kwargs):
        from pfund.datas.databoy import DataBoy
        
        self.__pfund_args__ = args
        self.__pfund_kwargs__ = kwargs
        
        # TODO: should NOT be class attributes, should be instance attributes, maybe treat class attributes as default values?
        cls = self.__class__
        cls.load_config()
        cls.load_params()
        
        self.name = self._get_default_name()
        self._run_mode: RunMode | None = None
        self._resolution: Resolution | None = None
        
        self.logger: logging.Logger | None = None
        self._logging_configurator: LoggingDictConfigurator | None = None
        self._engine: BaseEngine | EngineProxy | None = None
        self._consumer: Component | ActorProxy | None = None
        self._store: TradingStore | None = None
        self._databoy = DataBoy(self)
        # self._data_tool: BaseDataTool = self._create_data_tool()

        self.products: dict[str, BaseProduct] = {}
        self.models: dict[str, BaseModel | ActorProxy] = {}
        self.features: dict[str, BaseFeature | ActorProxy] = {}
        self.indicators: dict[str, BaseIndicator | ActorProxy] = {}

        # NOTE: current model's signal is consumer's prediction
        self.predictions = {}  # {model_name: pred_y}
        self._signals = {}  # {data: signal}, signal = output of predict()
        self._last_signal_ts = {}  # {data: ts}
        self._signal_cols = []
        self._num_signal_cols = 0

        # FIXME
        # self._is_ready = defaultdict(bool)  # {data: bool}
        self._is_running = False
        self._frozen = False
        self._assert_functions_signatures()
    
    # TODO: also check on_bar, on_tick, on_quote etc.
    def _assert_functions_signatures(self):
        pass

    @classmethod
    def load_config(cls, config: dict | None=None, file_path: str=''):
        if config:
            cls.config = config
        elif file_path:
            if config := load_yaml_file(file_path):
                cls.config = config
    
    @classmethod
    def load_params(cls, params: dict | None=None, file_path: str=''):
        if params:
            cls.params = params
        elif file_path:
            if params := load_yaml_file(file_path):
                cls.params = params
    
    def _freeze(self):
        self._frozen = True
    
    def is_frozen(self):
        return self._frozen
    
    def _assert_not_frozen(self):
        if self._frozen:
            raise RuntimeError(f"{self.name} is frozen, cannot modify it")
        
    # TODO: add versioning, run_id etc.
    def to_dict(self: Component) -> dict:
        metadata = {
            'class': self.__class__.__name__,
            'name': self.name,
            'component_type': self.component_type,
            'run_mode': self._run_mode,
            'config': self.config,
            'params': self.params,
            'datas': [repr(data) for data in self._databoy.datas.values()],
            'models': [model.to_dict() for model in self.models.values()],
            'features': [feature.to_dict() for feature in self.features.values()],
            'indicators': [indicator.to_dict() for indicator in self.indicators.values()],
            'signature': (self.__pfund_args__, self.__pfund_kwargs__),
            'data_signatures': self._databoy._data_signatures,
        }
        if self.is_strategy():
            metadata['strategies'] = [strategy.to_dict() for strategy in self.strategies.values()]
        if self.is_model():
            metadata['model'] = self.model
        return metadata

    @property
    def data_tool(self: Component) -> BaseDataTool:
        return self._data_tool
    dtl = data_tool
    
    def _create_data_tool(self: Component) -> BaseDataTool:
        import importlib
        from pfund.engines.trade_engine import TradeEngine
        data_tool = TradeEngine.data_tool
        DataTool: type[BaseDataTool] = getattr(importlib.import_module(f'pfund.data_tools.data_tool_{data_tool}'), f'{data_tool.value.capitalize()}DataTool')
        return DataTool()
    
    @property
    def databoy(self: Component) -> DataBoy:
        return self._databoy
    
    @property
    def datas(self: Component) -> dict[BaseProduct, dict[Resolution, TimeBasedData]]:
        return self._databoy.datas
    
    @property
    def store(self: Component) -> TradingStore:
        return self._store
    
    @property
    def components(self: Component) -> list[Component]:
        components = [*self.models.values(), *self.features.values(), *self.indicators.values()]
        if self.is_strategy():
            components.extend([*self.strategies.values()])
        return components
    
    @property
    def component_type(self) -> ComponentType:
        from pfund.strategies.strategy_base import BaseStrategy
        from pfund.indicators.indicator_base import BaseIndicator
        from pfund.features.feature_base import BaseFeature
        from pfund.models.model_base import BaseModel
        if isinstance(self, BaseStrategy):
            return ComponentType.strategy
        elif isinstance(self, BaseIndicator):
            return ComponentType.indicator
        elif isinstance(self, BaseFeature):
            return ComponentType.feature
        elif isinstance(self, BaseModel):
            return ComponentType.model
        
    def _setup_logging(self: Component, logging_configurator_or_config: LoggingDictConfigurator | dict):
        """Creates logger dynamically for component (e.g. strategy/model) using component's name.
        
        Args:
            logging_configurator_or_config: if run_mode == RunMode.REMOTE, LoggingDictConfigurator is passed in, otherwise logging_config (dict) is passed in
        """
        if self.is_remote():
            from pfund._logging.config import LoggingDictConfigurator
            # set up component's own logging when running in remote mode
            logging_config: dict = logging_configurator_or_config
            logging_configurator = LoggingDictConfigurator(logging_config)
            logging_configurator.configure()
        else:
            # reuse the same logging configurator as the one in the engine
            logging_configurator: LoggingDictConfigurator = logging_configurator_or_config
        logging_config = logging_configurator._pfund_config
        loggers_config = logging_config['loggers']
        default_logger_config = loggers_config[f'_{self.component_type}']
        logger_config = loggers_config.get(self.name, default_logger_config)
        logging_configurator.configure_logger(self.name, logger_config)
        self.logger = logging.getLogger(self.name)
        self._logging_configurator = logging_configurator
        self._databoy._set_logger(self.logger)
    
    def get_df(
        self: Component, 
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
        
    # NOTE: df = X + predictions generated by other strategies/models
    @property
    def df(self: Component):
        # TODO: self.store.load_data(...)
        return self.get_df(copy=False)

    @property
    def signals(self: Component):
        return self._signals
    
    @property
    def INDEX(self: Component):
        return self.data_tool.INDEX
    
    @property
    def GROUP(self: Component):
        return self.data_tool.GROUP

    @staticmethod
    def dt(ts: float) -> datetime.datetime:
        from pfund.utils.utils import convert_ts_to_dt
        return convert_ts_to_dt(ts)
    
    @staticmethod
    def now() -> datetime.datetime:
        return datetime.datetime.now(tz=datetime.timezone.utc)
    
    @staticmethod
    def get_delay(ts: float) -> float:
        return time.time() - ts
    
    def _set_engine(self, engine: BaseEngine | None):
        self._engine = EngineProxy(self._databoy) if engine is None else engine
    
    def _set_trading_store(self: Component, trading_store: TradingStore):
        self._store = trading_store
        self._store._set_logger(self.logger)

    @property
    def resolution(self: Component) -> Resolution | None:
        return self._resolution
    
    def _set_resolution(self, resolution: str):
        if self._resolution:
            raise ValueError(f"{self.name} already has a resolution {self._resolution}, cannot set to {resolution}")
        self._resolution = Resolution(resolution)

    def _set_name(self, name: str):
        if not name:
            return
        self.name = name
        if self.component_type not in self.name:
            self.name += f"_{self.component_type}"
    
    def _set_run_mode(self: Component, run_mode: RunMode):
        self._run_mode = run_mode

    @property
    def consumer(self: Component) -> Component | None:
        return self._consumer
    
    def _set_consumer(self: Component, consumer: BaseStrategy | BaseModel):
        if not self._consumer:
            self._consumer = consumer
        else:
            raise ValueError(f"{self.name} already has a consumer {self._consumer.name}")
    
    def is_strategy(self: Component) -> bool:
        return self.component_type == ComponentType.strategy
    
    def is_model(self: Component) -> bool:
        return self.component_type == ComponentType.model
    
    def is_indicator(self: Component) -> bool:
        return self.component_type == ComponentType.indicator
    
    def is_feature(self: Component) -> bool:
        return self.component_type == ComponentType.feature
    
    def is_running(self: Component):
        return self._is_running
    
    def is_remote(self: Component, relative: bool=True) -> bool:
        """
        Returns whether this component is running in a remote (Ray) process.

        Args:
            relative (bool): 
                - If True (default), only checks the component's own `_run_mode`.
                This reflects whether the component *itself* is declared to be remote.
                e.g. a model is running inside a strategy (ray actor), relatively the model is "local"
                - If False, walks up the `_consumer` chain to check if this component 
                is running *within* a remote context (e.g., inside a Ray actor).
                This captures whether the component is effectively remote due to being 
                nested inside another remote component.

        Returns:
            bool: True if the component (or any of its ancestors) is remote.
        """
        assert self._run_mode is not None, f"{self.name} has no run mode"
        if relative:
            return self._run_mode == RunMode.REMOTE
        else:
            consumer: Component | ActorProxy | None
            consumer = self._consumer
            while consumer is not None:
                if consumer.is_remote():
                    return True
                consumer = consumer._consumer
            return False
    
    def _add_product(self: Component, product_key: str, product: BaseProduct) -> BaseProduct:
        self._assert_not_frozen()
        if product_key not in self.products:
            self.products[product_key] = product
        else:
            existing_product = self.products[product_key]
            assert existing_product == product, f"key={product_key} is already used by {existing_product}, cannot use it for {product}"
        return self.products[product_key]
    
    def get_product(self: Component, product_key: str) -> BaseProduct:
        '''
        Args:
            product_key: product key is either product alias or product name
        '''
        return self.products[product_key]
    
    def get_data(self: Component, product: BaseProduct, resolution: str) -> TimeBasedData:
        data: TimeBasedData | None = self._databoy.get_data(product, resolution)
        if data is None:
            raise ValueError(f"data for {product} {resolution} not found")
        return data
    
    def _prepare_df(self: Component):
        return self.data_tool.prepare_df(ts_col_type='timestamp')
    
    def _append_to_df(self: Component, data: BaseData, **extra_data):
        return self.data_tool.append_to_df(data, self.predictions, **extra_data)
    
    def _get_default_name(self: Component):
        return self.__class__.__name__
    
    def get_default_signal_cols(self: Component, num_cols: int) -> list[str]:
        if num_cols == 1:
            columns = [self.name]
        else:
            columns = [f'{self.name}-{i}' for i in range(num_cols)]
        return columns
   
    def get_signal_cols(self: Component) -> list[str]:
        return self._signal_cols
    
    def _set_signal_cols(self: Component, columns: list[str]):
        self._signal_cols = [f'{self.name}-{col}' if not col.startswith(self.name) else col for col in columns]
        self._num_signal_cols = len(columns)
                
    def add_data(
        self, 
        trading_venue: tTRADING_VENUE,
        product: str,
        exchange: str='',
        symbol: str='',
        product_alias: str='',
        data_source: tDATA_SOURCE | None=None,
        data_origin: str='',
        data_config: DataConfigDict | DataConfig | None=None,
        **product_specs
    ) -> list[TimeBasedData]:
        '''
        Args:
            exchange: useful for TradFi brokers (e.g. IB), to specify the exchange (e.g. 'NASDAQ')
            symbol: useful for TradFi brokers (e.g. IB), to specify the symbol (e.g. 'AAPL')
            product: product basis, defined as {base_asset}_{quote_asset}_{product_type}, e.g. BTC_USDT_PERP
            product_alias: A short, user-defined identifier for the product.
                If not provided, the full product name (e.g., 'BTC_USDT_PERP') will be used by default.
                This is useful when the full product name is long and you prefer to reference the product using a shorter alias,
                such as 'BTC' instead of 'BTC_USDT_PERP'.
            product_specs: product specifications, e.g. expiration, strike_price etc.
        '''
        self._assert_not_frozen()
        data_source = data_source or trading_venue.upper()
        product: BaseProduct = self._engine._register_product(
            trading_venue=trading_venue,
            product_basis=product,
            exchange=exchange,
            symbol=symbol,
            product_alias=product_alias,
            **product_specs
        )
        self._add_product(product_alias or product.name, product)
        datas: list[TimeBasedData] = self._databoy.add_data(
            product=product,
            data_source=data_source,
            data_origin=data_origin,
            data_config=data_config
        )
        for data in datas:
            self._engine._register_market_data(self, data)
            # if not remote, needs its consumer to pass data to itself
            if not self.is_remote():
                consumer: Component | ActorProxy | None
                consumer = self._consumer
                while consumer is not None:
                    consumer.databoy._add_listener(self, data)
                    consumer = consumer._consumer
                    if consumer.is_remote():
                        break
        return datas
    
    # TODO
    def add_custom_data(self: Component):
        self._assert_not_frozen()
        raise NotImplementedError
    
    def _add_datas_from_consumer_if_none(self: Component) -> list[BaseData]:
        has_no_data = self._consumer and not self.datas
        if not has_no_data:
            return []
        self.logger.info(f"No data for {self.name}, adding datas from consumer {self._consumer.name}")
        datas: list[TimeBasedData] = self._consumer.list_datas()
        for data in datas:
            self._add_data(data)
            self._consumer._add_listener(self, data)
        return datas

    def get_orderbook(self: Component, product: BaseProduct) -> TimeBasedData | None:
        return self.databoy.get_data(product, '1q')
    
    def get_tradebook(self: Component, product: BaseProduct) -> TimeBasedData | None:
        return self.databoy.get_data(product, '1t')
    
    def _add_model_component(
        self: Component, 
        component: ModelT | FeatureT | IndicatorT,
        name: str='',
        min_data: int | None=None,
        max_data: int | None=None,
        group_data: bool=True,
        signal_cols: list[str] | None=None,
        ray_actor_options: dict | None=None,
        **ray_kwargs
    ) -> ModelT | FeatureT | IndicatorT | ActorProxy:
        '''Adds a model component to the current component.
        A model component is a model, feature, or indicator.
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

            ray_actor_options:
                Options for Ray actor.
                will be passed to ray actor like this: Actor.options(**ray_options).remote(**ray_kwargs)
        '''
        from pfund.utils.utils import derive_run_mode
        
        self._assert_not_frozen()

        Component = component.__class__
        ComponentName = Component.__name__
        component_type = component.component_type
        if component.is_model():
            from pfund.models.model_base import BaseModel
            components = self.models
            BaseClass = BaseModel
        elif component.is_feature():
            from pfund.features.feature_base import BaseFeature
            components = self.features
            BaseClass = BaseFeature
        elif component.is_indicator():
            from pfund.indicators.indicator_base import BaseIndicator
            components = self.indicators
            BaseClass = BaseIndicator
        else:
            raise ValueError(f"{component_type} '{ComponentName}' is not a model, feature, or indicator")
        assert isinstance(component, BaseClass), \
            f"{component_type} '{ComponentName}' is not an instance of {BaseClass.__name__}. Please create your {component_type} using 'class {ComponentName}(pf.{component_type.capitalize()})'"
        
        component_name = name or component.name
        if component_name in components:
            raise ValueError(f"{component_name} already exists")
        
        run_mode: RunMode = derive_run_mode(ray_kwargs)
        if is_remote := (run_mode == RunMode.REMOTE):
            component = ActorProxy(component, name=component_name, ray_actor_options=ray_actor_options, **ray_kwargs)
        component._set_name(component_name)
        component._set_run_mode(run_mode)
        component._set_resolution(self._resolution)
        # logging_configurator can't be serialized, so we need to pass in the original config instead in REMOTE mode
        component._setup_logging(self._logging_configurator._pfund_config if is_remote else self._logging_configurator)
        component._set_consumer(None if is_remote else self)
        component._set_engine(None if is_remote else self._engine)

        if min_data:
            component._set_min_data(min_data)
        if max_data:
            component._set_max_data(max_data)
        component._set_group_data(group_data)
        if signal_cols:
            component._set_signal_cols(signal_cols)
        
        components[component_name] = component
        self.logger.debug(f"added {component_name}")

        # NOTE: freeze the component when it's a local component but the consumer is a remote component
        # because the returned component is just a reference
        if self.is_remote() and not is_remote:
            component._freeze()
        return component
    
    def add_model(
        self: Component, 
        model: ModelT, 
        name: str='',
        min_data: int | None=None,
        max_data: int | None=None,
        group_data: bool=True,
        signal_cols: list[str] | None=None,
        ray_actor_options: dict | None=None,
        **ray_kwargs
    ) -> ModelT:
        return self._add_model_component(
            component=model,
            name=name,
            min_data=min_data,
            max_data=max_data,
            group_data=group_data,
            signal_cols=signal_cols,
            ray_actor_options=ray_actor_options,
            **ray_kwargs
        )
    
    def get_model(self, name: str) -> BaseModel | ActorProxy:
        return self.models[name]
    
    def add_feature(
        self: Component, 
        feature: FeatureT, 
        name: str='',
        min_data: int | None=None,
        max_data: int | None=None,
        group_data: bool=True,
        signal_cols: list[str] | None=None,
        ray_actor_options: dict | None=None,
        **ray_kwargs
    ) -> FeatureT:
        return self._add_model_component(
            component=feature, 
            name=name, 
            min_data=min_data, 
            max_data=max_data, 
            group_data=group_data,
            signal_cols=signal_cols,
            ray_actor_options=ray_actor_options,
            **ray_kwargs
        )
    
    def get_feature(self, name: str) -> BaseFeature | ActorProxy:
        return self.features[name]
    
    def add_indicator(
        self: Component, 
        indicator: IndicatorT, 
        name: str='',
        min_data: int | None=None,
        max_data: int | None=None,
        group_data: bool=True,
        signal_cols: list[str] | None=None,
        ray_actor_options: dict | None=None,
        **ray_kwargs
    ) -> IndicatorT:
        return self._add_model_component(
            component=indicator,
            name=name,
            min_data=min_data,
            max_data=max_data,
            group_data=group_data,
            signal_cols=signal_cols,
            ray_actor_options=ray_actor_options,
            **ray_kwargs
        )
    
    def get_indicator(self, name: str) -> BaseIndicator | ActorProxy:
        return self.indicators[name]

    def _on_quote(self: Component, data: QuoteData, **extra_data):
        product, bids, asks, ts = data.product, data.bids, data.asks, data.ts
        # TODO: wait for remote components' outputs
        local_components = self._local_components.get(data, [])
        for component in local_components:
            component._on_quote(data, **extra_data)
            self._update_outputs(data, component)
        # TODO: add to trading store, self._store
        self._append_to_df(data, **extra_data)
        self.on_quote(product, bids, asks, ts, **extra_data)

    def _on_tick(self: Component, data: TickData, **extra_data):
        product, px, qty, ts = data.product, data.px, data.qty, data.ts
        for listener in self._listeners[data]:
            listener._on_tick(data, **extra_data)
            self._update_outputs(data, listener)
        self._append_to_df(data, **extra_data)
        self.on_tick(product, px, qty, ts, **extra_data)
    
    def _on_bar(self: Component, data: BarData, **extra_data):
        product, bar, ts = data.product, data.bar, data.bar.end_ts
        for listener in self._listeners[data]:
            listener._on_bar(data, **extra_data)
            self._update_outputs(data, listener)
        self._append_to_df(data, **extra_data)
        self.on_bar(product, bar, ts, **extra_data)

    def _update_outputs(self: Component, data: BaseData, listener: BaseStrategy | BaseModel):
        pred_y: torch.Tensor | np.ndarray | None = listener._next(data)
        if pred_y is not None:
            signal_cols = listener.get_signal_cols()
            for i, col in enumerate(signal_cols):
                self.predictions[col] = pred_y[i]
    
    def start(self: Component):
        self._store._freeze()
        self._store._materialize()
        self._engine._register_component(
            consumer_name=self._consumer.name if self._consumer else None,
            component_name=self.name,
            component_metadata=self.to_dict(),
        )
        if not self.is_running():
            self.add_datas()
            self._add_datas_from_consumer_if_none()
            self.add_models()
            self.add_features()
            self.add_indicators()
            self._prepare_df()
            self.on_start()
            self._is_running = True
            self.logger.info(f"strategy '{self.name}' has started")
        else:
            self.logger.warning(f'strategy {self.name} has already started')
    
    def stop(self, reason: str=''):
        if self.is_running():
            self._is_running = False
            self.on_stop()
            if self._engine._use_ray:
                pass
            # TODO: stop components
            self.logger.info(f"strategy '{self.name}' has stopped, ({reason=})")
        else:
            self.logger.warning(f'strategy {self.name} has already stopped')

    
    '''
    ************************************************
    Override Methods
    Override these methods in your subclass to implement your custom behavior.
    ************************************************
    '''
    def on_quote(self, product, bids, asks, ts, **kwargs):
        raise NotImplementedError(f"Please define your own on_quote(product, bids, asks, ts, **kwargs) in your strategy '{self.name}'.")
    
    def on_tick(self, product, px, qty, ts, **kwargs):
        raise NotImplementedError(f"Please define your own on_tick(product, px, qty, ts, **kwargs) in your strategy '{self.name}'.")

    def on_bar(self, product, bar: Bar, ts, **kwargs):
        raise NotImplementedError(f"Please define your own on_bar(product, bar, ts, **kwargs) in your strategy '{self.name}'.")

    def add_datas(self: Component):
        pass
    
    def add_models(self: Component):
        pass

    def add_features(self: Component):
        pass
    
    def add_indicators(self: Component):
        pass
    
    def on_start(self: Component):
        pass
    
    def on_stop(self: Component):
        pass
    
    
    '''
    ************************************************
    Sugar Functions
    ************************************************
    '''
    def get_second_bar(self: Component, product: BaseProduct, period: int) -> BarData | None:
        return self.get_data(product, resolution=f'{period}s')
    
    def get_minute_bar(self: Component, product: BaseProduct, period: int) -> BarData | None:
        return self.get_data(product, resolution=f'{period}m')
    
    def get_hour_bar(self: Component, product: BaseProduct, period: int) -> BarData | None:
        return self.get_data(product, resolution=f'{period}h')
    
    def get_day_bar(self: Component, product: BaseProduct, period: int) -> BarData | None:
        return self.get_data(product, resolution=f'{period}d')
    