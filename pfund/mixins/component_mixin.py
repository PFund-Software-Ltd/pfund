from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import torch
    import numpy as np
    import pandas as pd
    import polars as pl
    from mtflow.stores.trading_store import TradingStore
    from pfeed.typing import tDataSource
    from pfund.datas.databoy import DataBoy
    from pfund.typing import (
        StrategyT, 
        ModelT, 
        IndicatorT, 
        FeatureT, 
        DataConfigDict, 
        tTradingVenue, 
        Component, 
        ComponentName,
    )
    from pfund.datas.data_bar import Bar
    from pfund.datas.data_base import BaseData
    from pfund.datas.data_time_based import TimeBasedData
    from pfund.products.product_base import BaseProduct
    from pfund.datas import QuoteData, TickData, BarData
    from pfund.engines.base_engine import BaseEngine
    from pfund.strategies.strategy_base import BaseStrategy
    from pfund.models.model_base import BaseModel
    from pfund.features.feature_base import BaseFeature
    from pfund.indicators.indicator_base import BaseIndicator
    from pfund.brokers.broker_base import BaseBroker
    from pfund.engines.base_engine_settings import BaseEngineSettings

import time
import logging
import datetime

from pfund._logging.buffer import LogBuffer
from pfund.datas.resolution import Resolution
from pfund.datas.data_config import DataConfig
from pfund.proxies.actor_proxy import ActorProxy
from pfund.utils.utils import load_yaml_file
from pfund.enums import ComponentType, RunMode, Environment, Broker, TradingVenue

    
class ComponentMixin:
    config = {}
    params = {}

    # custom post init for-common attributes of strategy and model
    def __mixin_post_init__(self: Component, *args, **kwargs):
        from pfund.datas.databoy import DataBoy

        self.__pfund_args__ = args
        self.__pfund_kwargs__ = kwargs
        
        # TODO: should NOT be class attributes, should be instance attributes, maybe treat class attributes as default values?
        cls = self.__class__
        cls.load_config()
        cls.load_params()
        
        self._env: Environment | None = None
        self.name = self._get_default_name()
        self._run_mode: RunMode | None = None
        self._resolution: Resolution | None = None
        
        self.logger: logging.Logger | None = logging.getLogger('pfund')
        self._proxy: ActorProxy | None = None
        self._engine: BaseEngine | None = None
        self._settings: BaseEngineSettings | None = None
        self._logging_config: dict | None = None
        self._consumers: list[Component | ActorProxy] = []
        self._store: TradingStore | None = None
        self._databoy = DataBoy(self)

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
        self._is_gathered = False
        self._assert_functions_signatures()
    
    # TODO: also check on_bar, on_tick, on_quote etc.
    def _assert_functions_signatures(self):
        pass

    def _hydrate(
        self: Component,
        env: Environment | str,
        name: ComponentName,
        run_mode: RunMode,
        resolution: Resolution | str,
        engine: BaseEngine | None,
        settings: BaseEngineSettings,
        logging_config: dict,
    ):
        """
        Hydrates the component with necessary attributes after initialization.
        
        This method sets up essential properties like name, run mode, resolution, engine, and settings
        which are not predefined in the constructor to improve developer experience (DX) for users
        creating custom component classes (strategies, models, features, etc.).
        
        Args:
            name (ComponentName): The name to assign to this component.
            run_mode (RunMode): The mode in which the component will run (e.g., local or remote).
            resolution (Resolution | str): The data resolution used by this component.
            engine (BaseEngine | None): The engine instance associated with this component. It is None if the component is running in a remote process.
            settings (BaseEngineSettings): Configuration settings for the engine and component.
            logging_config (dict): Configuration for logging.
        """
        self._env = Environment[env.upper()]
        self._set_name(name)
        self._set_resolution(resolution)
        self._run_mode = run_mode
        self._engine = engine
        self._settings = settings
        if self.is_remote():
            self._setup_logging(logging_config)
        if not self.is_wasm():
            self.databoy._setup_messaging()

    def _setup_logging(self: Component, logging_config: dict) -> dict[str, int]:
        '''Sets up logging for component running in remote process, uses zmq's PUBHandler to send logs to engine'''
        from zmq.log.handlers import PUBHandler
        from pfund.zeromq import ZeroMQ
        from pfund._logging.config import LoggingDictConfigurator
        from pfund.utils.utils import get_free_port

        # configure logging based on pfund's logging config, e.g. log_level, log_file, log_format, etc.
        logging_configurator = LoggingDictConfigurator(logging_config)
        logging_configurator.configure()
        self._logging_config = logging_config
        self.logger = logging.getLogger('pfund')
        
        # remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
            handler.close()
            
        # TODO: create a subclass of PUBHandler that can standardize the msg and support log buffer
        # add zmq PUBhandler
        zmq_url = self._settings.zmq_urls.get(self.name, ZeroMQ.DEFAULT_URL)
        zmq_port = get_free_port()
        zmq_handler = PUBHandler(f'{zmq_url}:{zmq_port}')
        self.logger.addHandler(zmq_handler)
        self.databoy._update_zmq_ports_in_use(
            {self.name+'_logger': zmq_port}
        )
    
    def _get_zmq_ports_in_use(self) -> dict[str, int]:
        return self.databoy._get_zmq_ports_in_use()
    
    def _set_proxy(self, proxy: ActorProxy):
        self._proxy = proxy
    
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
    
    # TODO: add versioning, run_id etc.
    def to_dict(self: Component, stringify: bool=True) -> dict:
        '''
        Args:
            stringify: if True, serializes the data to a string, otherwise returns the data object
        '''
        metadata = {
            'class': self.__class__.__name__,
            'name': self.name,
            'component_type': self.component_type.value,
            'run_mode': self._run_mode.value,
            'config': self.config,
            'params': self.params,
            'datas': [repr(data) if stringify else data for data in self.databoy.get_datas()],
            'models': [model.to_dict(stringify=stringify) for model in self.models.values()],
            'features': [feature.to_dict(stringify=stringify) for feature in self.features.values()],
            'indicators': [indicator.to_dict(stringify=stringify) for indicator in self.indicators.values()],
            'signature': (self.__pfund_args__, self.__pfund_kwargs__),
            'data_signatures': self.databoy._data_signatures,
        }
        if self.is_strategy():
            metadata['strategies'] = [strategy.to_dict(stringify=stringify) for strategy in self.strategies.values()]
            metadata['accounts'] = [repr(account) if stringify else account for account in self.accounts.values()]
        elif self.is_model():
            metadata['model'] = self.model.__class__.__name__
        if not stringify:
            for data in metadata['datas']:
                # removes non-serializable attributes
                if data.is_quote():
                    data._orderbook = None
        return metadata
    
    @property
    def env(self: Component) -> Environment:
        return self._env
    
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
    def components(self: Component) -> list[Component | ActorProxy]:
        components = [*self.models.values(), *self.features.values(), *self.indicators.values()]
        if self.is_strategy():
            components.extend([*self.strategies.values()])
        return components
    
    @property
    def consumers(self: Component) -> list[Component | ActorProxy]:
        return self._consumers
    
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
    
    def _set_trading_store(self: Component, trading_store: TradingStore):
        self._store = trading_store

    @property
    def resolution(self: Component) -> Resolution | None:
        return self._resolution
    
    def _set_resolution(self, resolution: Resolution | str):
        if self._resolution:
            raise ValueError(f"{self.name} already has a resolution {self._resolution}, cannot set to {resolution}")
        self._resolution = Resolution(resolution)

    def _set_name(self, name: str):
        if not name:
            return
        self.name = name
        if self.component_type not in self.name.lower():
            self.name += f"_{self.component_type}"
    
    def _add_consumer(self: Component, consumer: Component | ActorProxy):
        if consumer not in self._consumers:
            self._consumers.append(consumer)
        else:
            raise ValueError(f"{self.name} already has a consumer {consumer}")
    
    def is_strategy(self: Component) -> bool:
        return self.component_type == ComponentType.strategy
    
    def is_model(self: Component) -> bool:
        return self.component_type == ComponentType.model
    
    def is_indicator(self: Component) -> bool:
        return self.component_type == ComponentType.indicator
    
    def is_feature(self: Component) -> bool:
        return self.component_type == ComponentType.feature
    
    def is_running(self: Component) -> bool:
        return self._is_running
    
    def is_remote(self: Component, direct_only: bool=True) -> bool:
        """
        Returns whether this component is running in a remote (Ray) process.

        Args:
            direct_only (bool): 
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
        is_remote = self._run_mode == RunMode.REMOTE
        if is_remote or direct_only:
            return is_remote
        # if not remote, walk up the consumer chain to check if any of the ancestors is remote
        for consumer in self._consumers:
            if consumer.is_remote(direct_only=direct_only):
                return True
        return False
    
    def is_wasm(self: Component) -> bool:
        return self._run_mode == RunMode.WASM
    
    def _create_product(
        self,
        trading_venue: tTradingVenue,
        basis: str,
        exch: str='',
        symbol: str='',
        name: str='',
        **specs
    ) -> BaseProduct:
        from pfund.brokers import create_broker
        # NOTE: broker is only used to create product but nothing else
        broker = create_broker(env=self._env, bkr=TradingVenue[trading_venue.upper()].broker)
        if broker.name == Broker.CRYPTO:
            exch = trading_venue
            product = broker.add_product(exch=exch, basis=basis, name=name, symbol=symbol, **specs)
        elif broker.name == Broker.IB:
            product = broker.add_product(exch=exch, basis=basis, name=name, symbol=symbol, **specs)
        else:
            raise NotImplementedError(f"Broker {broker.name} is not supported")
        return product
    
    def _add_product(self: Component, product: BaseProduct) -> BaseProduct:
        if product.name not in self.products:
            self.products[product.name] = product
        else:
            existing_product = self.products[product.name]
            assert existing_product == product, f"{product.name=} is already used by {existing_product}, cannot use it for {product}"
        return self.products[product.name]
    
    def get_product(self: Component, name: str) -> BaseProduct:
        return self.products[name]
    
    def get_data(self: Component, product: BaseProduct, resolution: str) -> TimeBasedData:
        data: TimeBasedData | None = self.databoy.get_data(product, resolution)
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
        trading_venue: tTradingVenue,
        product: str,
        exchange: str='',
        symbol: str='',
        product_name: str='',
        data_source: tDataSource | None=None,
        data_origin: str='',
        data_config: DataConfigDict | DataConfig | None=None,
        **product_specs
    ) -> list[TimeBasedData]:
        '''
        Args:
            exchange: useful for TradFi brokers (e.g. IB), to specify the exchange (e.g. 'NASDAQ')
            symbol: useful for TradFi brokers (e.g. IB), to specify the symbol (e.g. 'AAPL')
            product: product basis, defined as {base_asset}_{quote_asset}_{product_type}, e.g. BTC_USDT_PERP
            product_name: A user-defined identifier for the product.
                If not provided, the default product symbol (e.g. 'BTC_USDT_PERP', 'TSLA241213C00075000') will be used.
                This is useful when you need to distinguish between similar instruments, such as options 
                with different strike prices and expiration dates. Instead of using long identifiers like 
                'BTC_USDT_OPTION_100000_20250620' and 'BTC_USDT_OPTION_101000_20250920', you can assign 
                simpler names like 'BTC_OPT1' and 'BTC_OPT2'.
                Note:
                    It is the user's responsibility to manage and maintain these custom product names.
            product_specs: product specifications, e.g. expiration, strike_price etc.
        '''
        product: BaseProduct = self._create_product(
            trading_venue=trading_venue,
            basis=product,
            exch=exchange,
            symbol=symbol,
            name=product_name,
            **product_specs
        )
        self._add_product(product)
        datas: list[TimeBasedData] = self.databoy.add_data(
            product=product,
            data_source=data_source or trading_venue,
            data_origin=data_origin,
            data_config=data_config
        )
        return datas
    
    # TODO
    def add_custom_data(self: Component):
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
    
    def _add_component(
        self: Component, 
        component: StrategyT | ModelT | FeatureT | IndicatorT | ActorProxy,
        name: str='', 
        min_data: int | None=None,
        max_data: int | None=None,
        group_data: bool=True,
        signal_cols: list[str] | None=None,
        ray_actor_options: dict | None=None,
        **ray_kwargs
    ) -> StrategyT | ModelT | FeatureT | IndicatorT | ActorProxy:
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

            ray_kwargs: kwargs for ray actor, e.g. num_cpus
            ray_actor_options:
                Options for Ray actor.
                will be passed to ray actor like this: Actor.options(**ray_options).remote(**ray_kwargs)
        '''
        from pfund.utils.utils import derive_run_mode

        Component = component.__class__
        ComponentName = Component.__name__
        if component.is_strategy():
            assert self.component_type == ComponentType.strategy, \
                f"cannot add strategy '{ComponentName}' to {self.component_type} '{self.name}'"
            components = self.strategies
            BaseClass = BaseStrategy
        elif component.is_model():
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
            raise ValueError(f"{component.component_type} '{ComponentName}' is not a model, feature, or indicator")

        if not isinstance(component, ActorProxy):
            component_type = component.component_type
            assert isinstance(component, BaseClass), \
                f"{component_type} '{ComponentName}' is not an instance of {BaseClass.__name__}. Please create your {component_type} using 'class {ComponentName}(pf.{component_type.capitalize()})'"
            component_name = name or component.name
            if component_name in components:
                raise ValueError(f"{component_name} already exists")
            
            run_mode: RunMode = derive_run_mode(ray_kwargs)
            if is_remote := (run_mode == RunMode.REMOTE):
                component = ActorProxy(component, name=component_name, ray_actor_options=ray_actor_options, **ray_kwargs)
                component._set_proxy(component)
            component._hydrate(
                env=self._env,
                name=component_name,
                run_mode=run_mode,
                resolution=self._resolution,
                engine=self._engine,
                settings=self._settings,
                logging_config=self._logging_config,
            )
            
            if signal_cols:
                component._set_signal_cols(signal_cols)

            # FIXME: check if min_data, max_data and group_data are needed when component_type is strategy
            if not component.is_strategy():
                if min_data:
                    component._set_min_data(min_data)
                if max_data:
                    component._set_max_data(max_data)
                component._set_group_data(group_data)
        else:
            is_remote = True
        component._add_consumer(self._proxy if is_remote else self)
        components[component.name] = component
        self.logger.debug(f"added {component.name}")
    
        if self.is_remote() and not is_remote:
            # NOTE: returns None when adding a local component to a remote component to avoid returning a serialized (copied) object
            return None
        else:
            return component
    
    def add_model(
        self: Component, 
        model: ModelT | ActorProxy,
        name: str='',
        min_data: int | None=None,
        max_data: int | None=None,
        group_data: bool=True,
        signal_cols: list[str] | None=None,
        ray_actor_options: dict | None=None,
        **ray_kwargs
    ) -> ModelT | ActorProxy:
        return self._add_component(
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
        feature: FeatureT | ActorProxy, 
        name: str='',
        min_data: int | None=None,
        max_data: int | None=None,
        group_data: bool=True,
        signal_cols: list[str] | None=None,
        ray_actor_options: dict | None=None,
        **ray_kwargs
    ) -> FeatureT | ActorProxy:
        return self._add_component(
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
        indicator: IndicatorT | ActorProxy,
        name: str='',
        min_data: int | None=None,
        max_data: int | None=None,
        group_data: bool=True,
        signal_cols: list[str] | None=None,
        ray_actor_options: dict | None=None,
        **ray_kwargs
    ) -> IndicatorT | ActorProxy:
        return self._add_component(
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
    
    def _gather(self: Component):
        '''Sets up everything before start'''
        # NOTE: use is_gathered to avoid a component being gathered multiple times when it's a shared component
        if not self._is_gathered:
            self._is_gathered = True
            self.add_datas()
            if self.is_strategy():
                self.add_strategies()
            self.add_models()
            self.add_features()
            self.add_indicators()
            # all components are gathered/added, all zmq ports relevant to this component are finalized, can subscribe now
            self.databoy.subscribe()
            # TODO:
            # self._add_datas_from_consumer_if_none()
            # TODO:
            # self._store._freeze()
            # self._store._materialize()
            # self._prepare_df()
            for component in self.components:
                component._gather()
            self.logger.info(f"'{self.name}' has gathered")
        else:
            self.logger.info(f"'{self.name}' has already gathered")
    
    def start(self: Component):
        if not self.is_running():
            self._is_running = True
            self.on_start()
            self.databoy.start()
            for component in self.components:
                component.start()
            self.logger.info(f"'{self.name}' has started")
        else:
            self.logger.info(f"'{self.name}' has already started")
    
    def stop(self, reason: str=''):
        '''Stops the component, keep the internal states'''
        if self.is_running():
            self._is_running = False
            self.on_stop()
            self.databoy.stop()
            for component in self.components:
                component.stop()
            self.logger.info(f"'{self.name}' has stopped, ({reason=})")
        else:
            self.logger.info(f"'{self.name}' has already stopped")

    
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
    