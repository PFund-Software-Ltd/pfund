# pyright: reportUninitializedInstanceVariable=false
from __future__ import annotations
from typing import TYPE_CHECKING, Any, cast
if TYPE_CHECKING:
    import torch
    import numpy as np
    import pandas as pd
    import polars as pl
    from pfund.typing import (
        ComponentT, 
        ModelT, 
        IndicatorT, 
        FeatureT, 
        Component,
        ComponentName,
        ProductName,
        ResolutionRepr,
    )
    from pfund.engines.engine_context import EngineContext
    from pfund.datas.data_market import MarketData
    from pfund.datas.data_bar import Bar
    from pfund.datas.data_base import BaseData
    from pfund.datas.data_time_based import TimeBasedData
    from pfund.entities.products.product_base import BaseProduct
    from pfund.datas import QuoteData, TickData, BarData
    from pfund.engines.settings.trade_engine_settings import TradeEngineSettings
    from pfund.components.strategies.strategy_base import BaseStrategy
    from pfund.components.models.model_base import BaseModel
    from pfund.components.features.feature_base import BaseFeature
    from pfund.components.indicators.indicator_base import BaseIndicator
    from pfund.datas.stores.trading_store import TradingStore
    from pfund.brokers.broker_base import BaseBroker

import time
import importlib
import logging
import datetime

from pfund_kit.utils import yaml
from pfeed.enums import DataSource
from pfeed.storages.storage_config import StorageConfig
from pfund.datas.resolution import Resolution, ResolutionUnit
from pfund.datas.data_config import DataConfig
from pfund.components.actor_proxy import ActorProxy
from pfund.enums import ComponentType, RunMode, Environment, Broker, TradingVenue

    
class ComponentMixin:
    config = {}
    params = {}

    # custom post init for-common attributes of strategy and model
    def __mixin_post_init__(self, *args: Any, **kwargs: Any):
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
        
        self.logger: logging.Logger = logging.getLogger('pfund')
        self._proxy: ActorProxy[Component] | None = None
        self._context: EngineContext | None = None
        self._consumers: list[Component | ActorProxy[Component]] = []
        self._store: TradingStore | None = None
        self.databoy = DataBoy(self)

        self.products: dict[ProductName, BaseProduct] = {}
        self.models: dict[str, BaseModel | ActorProxy[BaseModel]] = {}
        self.features: dict[str, BaseFeature | ActorProxy[BaseFeature]] = {}
        self.indicators: dict[str, BaseIndicator | ActorProxy[BaseIndicator]] = {}

        # FIXME: to be removed, should return dict with keys as column names in predict()
        # otherwise, use default cols
        self._signal_cols: list[str] = []
        

        # FIXME
        # self._is_ready = defaultdict(bool)  # {data: bool}
        self._is_running = False
        self._is_gathered = False
        self._assert_functions_signatures()
    
    @property
    def env(self) -> Environment:
        assert self._context is not None, 'context is not set'
        return self._context.env
    
    @property
    def run_mode(self):
        return self._run_mode
    
    @property
    def context(self) -> EngineContext:
        assert self._context is not None, 'context is not set'
        return self._context
    
    @property
    def settings(self) -> TradeEngineSettings:
        return cast(TradeEngineSettings, self.context.settings)
    
    @property
    def store(self) -> TradingStore:
        assert self._store is not None, 'store is not set'
        return self._store
    
    # TODO: also check on_bar, on_tick, on_quote etc.
    def _assert_functions_signatures(self):
        pass

    def _hydrate(
        self,
        name: ComponentName,
        run_mode: RunMode,
        resolution: Resolution | str,
        engine_context: EngineContext,
    ):
        """
        Hydrates the component with necessary attributes after initialization.
        
        Args:
            name (ComponentName): The name to assign to this component.
            run_mode (RunMode): The mode in which the component will run (e.g., local or remote).
            resolution (Resolution | str): The data resolution used by this component.
            engine_context (EngineContext): The engine context associated with this component. It is None if the component is running in a remote process.
        """
        from pfund.datas.stores.trading_store import TradingStore
        self._context = engine_context
        self._run_mode = run_mode
        self._set_name(name)
        self._set_resolution(resolution)
        if self.is_remote():
            self._setup_logging()
        self._store = TradingStore(engine_context)
        if self.env != Environment.BACKTEST:
            self.databoy._setup_messaging()
    
    # FIXME: integrate pfund_kit logging instead
    def _setup_logging(self):
        '''Sets up logging for component running in remote process, uses zmq's PUBHandler to send logs to engine'''
        from pfund.utils.zmq_pub_handler import ZMQPubHandler
        from pfeed.streaming.zeromq import ZeroMQ
        from pfund.logging.config import LoggingDictConfigurator
        from pfund_kit.utils import get_free_port

        # configure logging based on pfund's logging config, e.g. log_level, log_file, log_format, etc.
        logging_configurator = LoggingDictConfigurator(self._context.logging_config)
        logging_configurator.configure()
        
        # remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
            handler.close()
            
        # add zmq PUBhandler
        zmq_url = self._context.settings.zmq_urls.get(self.name, ZeroMQ.DEFAULT_URL)
        zmq_port = get_free_port()
        zmq_handler = ZMQPubHandler(f'{zmq_url}:{zmq_port}')
        zmq_formatter = logging.Formatter(
            fmt='%(message)s | from:%(filename)s fn:%(funcName)s ln:%(lineno)d (sent@%(asctime)s.%(msecs)03d)',
            datefmt='%H:%M:%S'
        )
        zmq_handler.setFormatter(zmq_formatter)
        self.logger.addHandler(zmq_handler)
        self.databoy._update_zmq_ports_in_use(
            {self.name + '_logger': zmq_port}
        )
    
    def _get_zmq_ports_in_use(self) -> dict[str, int]:
        return self.databoy._get_zmq_ports_in_use()
    
    def _get_datas_in_use(self) -> list[BaseData]:
        '''
        Gets all datas in use by this component, including the datas from its components.
        Since data objects are created by databoy, engine has no access to them.
        This method is used to return data objects to engine for registration.
        '''
        datas = self.databoy.get_datas()
        for component in self.components:
            datas.extend(component._get_datas_in_use())
        return list(set(datas))
    
    def _set_proxy(self, proxy: ActorProxy[Component]):
        self._proxy = proxy
    
    @classmethod
    def load_config(cls, config: dict | None=None, file_path: str=''):
        if config:
            cls.config = config
        elif file_path:
            if config := yaml.load(file_path):
                cls.config = config
    
    @classmethod
    def load_params(cls, params: dict | None=None, file_path: str=''):
        if params:
            cls.params = params
        elif file_path:
            if params := yaml.load(file_path):
                cls.params = params
    
    # TODO: add versioning, run_id etc.
    # TODO: create ComponentMetadata class (typeddataclass/pydantic model)
    def to_dict(self) -> dict:
        metadata = {
            'class': self.__class__.__name__,
            'env': self.env.value,
            'name': self.name,
            'component_type': self.component_type.value,
            'run_mode': self.run_mode.value,
            'config': self.config,
            'params': self.params,
            'consumers': [consumer.name for consumer in self._consumers],
            'datas': [data.to_dict() for data in self.databoy.get_datas()],
            'models': [model.to_dict() for model in self.models.values()],
            'features': [feature.to_dict() for feature in self.features.values()],
            'indicators': [indicator.to_dict() for indicator in self.indicators.values()],
            'signature': (self.__pfund_args__, self.__pfund_kwargs__),
            'data_signatures': self.databoy._data_signatures,
        }
        return metadata
    
    def get_default_signal_cols(self, num_cols: int) -> list[str]:
        if num_cols == 1:
            columns = [self.name]
        else:
            columns = [f'{self.name}-{i}' for i in range(num_cols)]
        return columns
   
    def _set_signal_cols(self, columns: list[str]):
        self._signal_cols = [f'{self.name}-{col}' if not col.startswith(self.name) else col for col in columns]
    
    @property
    def datas(self) -> dict[BaseProduct, dict[Resolution, TimeBasedData]]:
        return self.databoy.datas
    
    @property
    def components(self) -> list[Component | ActorProxy[Component]]:
        components = [*self.models.values(), *self.features.values(), *self.indicators.values()]
        if self.is_strategy():
            components.extend([*self.strategies.values()])
        return components
    
    @property
    def consumers(self) -> list[Component | ActorProxy[Component]]:
        return self._consumers
    
    @property
    def component_type(self) -> ComponentType:
        from pfund.components.strategies.strategy_base import BaseStrategy
        from pfund.components.indicators.indicator_base import BaseIndicator
        from pfund.components.features.feature_base import BaseFeature
        from pfund.components.models.model_base import BaseModel
        if isinstance(self, BaseStrategy):
            return ComponentType.strategy
        elif isinstance(self, BaseIndicator):
            return ComponentType.indicator
        elif isinstance(self, BaseFeature):
            return ComponentType.feature
        elif isinstance(self, BaseModel):
            return ComponentType.model
    
    def get_df(
        self, 
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
    def df(self):
        # TODO: self.store.load_data(...)
        return self.get_df(copy=False)

    @property
    def INDEX(self):
        return self.data_tool.INDEX
    
    @property
    def GROUP(self):
        return self.data_tool.GROUP

    @staticmethod
    def dt(ts: float) -> datetime.datetime:
        from pfund_kit.utils.temporal import convert_ts_to_dt
        return convert_ts_to_dt(ts)
    
    @staticmethod
    def now() -> datetime.datetime:
        return datetime.datetime.now(tz=datetime.timezone.utc)
    
    @staticmethod
    def get_delay(ts: float) -> float:
        return time.time() - ts
    
    @property
    def resolution(self) -> Resolution | None:
        return self._resolution
    
    def _set_resolution(self, resolution: Resolution | str):
        if self._resolution:
            raise ValueError(f"{self.name} already has a resolution {self._resolution}, cannot set to {resolution}")
        resolution = Resolution(resolution)
        if not resolution.is_bar():
            raise ValueError(f"{self.component_type} must use a bar resolution (e.g. '1s', '1m', '1h', '1d'), got {resolution=}")
        self._resolution = resolution

    def _set_name(self, name: str):
        if not name:
            return
        self.name = name
        if not self.name.lower().endswith(self.component_type):
            self.name += f"_{self.component_type}"
    
    def _add_consumer(self, consumer: Component | ActorProxy[Component]):
        if consumer not in self._consumers:
            self._consumers.append(consumer)
        else:
            raise ValueError(f"{self.name} already has a consumer {consumer}")
    
    def is_strategy(self) -> bool:
        return self.component_type == ComponentType.strategy
    
    def is_model(self) -> bool:
        return self.component_type == ComponentType.model
    
    def is_indicator(self) -> bool:
        return self.component_type == ComponentType.indicator
    
    def is_feature(self) -> bool:
        return self.component_type == ComponentType.feature
    
    def is_running(self) -> bool:
        return self._is_running
    
    def is_remote(self, direct_only: bool=True) -> bool:
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
        assert self.run_mode is not None, f"{self.name} has no run mode"
        is_remote = self.run_mode == RunMode.REMOTE
        if is_remote or direct_only:
            return is_remote
        # if not remote, walk up the consumer chain to check if any of the ancestors is remote
        for consumer in self._consumers:
            if consumer.is_remote(direct_only=direct_only):
                return True
        return False
    
    def _add_product(
        self,
        tv: TradingVenue | str,
        basis: str,
        exch: str='',
        symbol: str='',
        name: str='',
        **specs: Any
    ) -> BaseProduct:
        from pfund.brokers import create_broker
        # NOTE: broker is only used to create product but nothing else
        broker: BaseBroker = create_broker(env=self.env, bkr=TradingVenue[tv.upper()].broker, settings=self.settings)
        if broker.name == Broker.CRYPTO:
            exch = tv
            product: BaseProduct = broker.add_product(exch=exch, basis=basis, name=name, symbol=symbol, **specs)
        elif broker.name == Broker.IBKR:
            product: BaseProduct = broker.add_product(exch=exch, basis=basis, name=name, symbol=symbol, **specs)
        else:
            raise NotImplementedError(f"Broker {broker.name} is not supported")
        if product.name not in self.products:
            self.products[product.name] = product
        else:
            existing_product = self.products[product.name]
            assert existing_product == product, f"{product.name=} is already used by {existing_product}, cannot use it for {product}"
        return self.products[product.name]
    
    def get_product(self, name: ProductName) -> BaseProduct | None:
        return self.products.get(name, None)
    
    def get_data(self, product: ProductName, resolution: ResolutionRepr) -> MarketData | None:
        return self.databoy.get_data(product, resolution)
    
    def _append_to_df(self, data: BaseData):
        return self.data_tool.append_to_df(data, self.predictions)
    
    def _get_default_name(self):
        return self.__class__.__name__
    
    @staticmethod
    def _get_supported_resolutions(product: BaseProduct) -> dict[ResolutionUnit, list[int]]:
        supported_resolutions: dict[ResolutionUnit, list[int]]
        if product.bkr == Broker.CRYPTO:
            Exchange = getattr(importlib.import_module(f'pfund.brokers.crypto.exchanges.{product.exch.lower()}.exchange'), 'Exchange')
            supported_resolutions = Exchange.get_supported_resolutions(product)
        elif product.bkr == Broker.IBKR:
            InteractiveBrokersAPI = getattr(importlib.import_module('pfund.brokers.ibkr.api'), 'InteractiveBrokersAPI')
            supported_resolutions = InteractiveBrokersAPI.SUPPORTED_RESOLUTIONS
        else:
            raise NotImplementedError(f'broker {product.bkr} is not supported')
        return supported_resolutions
    
    def _resolve_data_config(self, product: BaseProduct, data_config: DataConfig | None) -> DataConfig:
        '''Resolves and configures DataConfig with defaults and automatic settings.

        Sets primary resolution from component resolution, derives data source from
        product's trading venue, and configures resampling for extra resolutions.

        Args:
            product: The product to configure data for
            data_config: Optional data configuration (creates default if None)

        Returns:
            Fully configured DataConfig ready for data subscription

        Raises:
            AssertionError: If component resolution is not set
            ValueError: If data_source cannot be derived from product's trading venue
        '''
        data_config = data_config or DataConfig()
        # set data config's primary resolution to be the component's resolution
        assert self.resolution is not None, 'component resolution is not set'
        data_config.primary_resolution = self.resolution
        # derive data_source from trading_venue if not provided
        if data_config.data_source is None:
            if product.tv in DataSource.__members__:
                data_config.data_source = DataSource[product.tv]
            else:
                raise ValueError("data_source must be provided")
        data_config = self._auto_resample_data_config(product, data_config)
        # TODO: auto shift data config
        data_config = self._auto_shift_data_config(data_config)
        return data_config
    
    def _auto_resample_data_config(self, product: BaseProduct, data_config: DataConfig) -> DataConfig:
        supported_resolutions = self._get_supported_resolutions(product)
        original_resample = data_config.resample.copy()
        is_auto_resampled = data_config.auto_resample(supported_resolutions)
        if is_auto_resampled:
            self.logger.warning(
                f'{product.name} resolution={repr(data_config.primary_resolution)} extra_resolutions={data_config.extra_resolutions} ' +
                f' data is auto-resampled from {original_resample} to {data_config.resample}'
            )
        return data_config
    
    # TODO: detect bar shift based on the returned data by e.g. Yahoo Finance, its hourly data starts from 9:30 to 10:30 etc.
    def _auto_shift_data_config(self, data_config: DataConfig) -> DataConfig:
        # data_config.auto_shift()
        return data_config
    
    def add_data(
        self, 
        trading_venue: TradingVenue | str,
        product: str,
        exchange: str='',
        symbol: str='',
        product_name: str='',
        data_config: DataConfig | None=None,
        storage_config: StorageConfig | None=None,
        **product_specs: Any
    ) -> list[MarketData]:
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
        product: BaseProduct = self._add_product(
            tv=trading_venue,
            basis=product,
            exch=exchange,
            symbol=symbol,
            name=product_name,
            **product_specs
        )
        datas: list[MarketData] = self.databoy.add_data(
            product=product, 
            data_config=self._resolve_data_config(product, data_config)
        )
        for data in datas:
            if not data.is_bar():
                continue
            self.store.market.add_data(data=data, storage_config=storage_config)
        return datas
    
    # TODO
    def add_custom_data(self):
        raise NotImplementedError
    
    def _add_datas_from_consumer_if_none(self) -> list[BaseData]:
        has_no_data = self._consumer and not self.datas
        if not has_no_data:
            return []
        self.logger.info(f"No data for {self.name}, adding datas from consumer {self._consumer.name}")
        datas: list[TimeBasedData] = self._consumer.list_datas()
        for data in datas:
            self._add_data(data)
            self._consumer._add_listener(self, data)
        return datas

    def get_orderbook(self, product: BaseProduct) -> TimeBasedData | None:
        return self.get_data(product, '1q')
    
    def get_tradebook(self, product: BaseProduct) -> TimeBasedData | None:
        return self.get_data(product, '1t')
    
    def _add_component(
        self, 
        component: ComponentT | ActorProxy[ComponentT],
        name: str='', 
        min_data: int | None=None,
        max_data: int | None=None,
        group_data: bool=True,
        signal_cols: list[str] | None=None,
        ray_actor_options: dict | None=None,
        **ray_kwargs
    ) -> ComponentT | ActorProxy[ComponentT] | None:
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
        Component = component.__class__
        ComponentName = Component.__name__
        if component.is_model():
            from pfund.components.models.model_base import BaseModel
            components = self.models
            BaseClass = BaseModel
        elif component.is_feature():
            from pfund.components.features.feature_base import BaseFeature
            components = self.features
            BaseClass = BaseFeature
        elif component.is_indicator():
            from pfund.components.indicators.indicator_base import BaseIndicator
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
            
            if ray_kwargs:
                if not self.is_remote(direct_only=False):
                    from pfeed.utils.ray import setup_ray
                    setup_ray()
                component = ActorProxy(component, name=component_name, ray_actor_options=ray_actor_options, **ray_kwargs)
                component._set_proxy(component)
            component._hydrate(
                name=component_name,
                run_mode=RunMode.REMOTE if ray_kwargs else RunMode.LOCAL,
                resolution=self._resolution,
                engine=self._context,
            )
            
            if signal_cols:
                component._set_signal_cols(signal_cols)

            # FIXME: check if min_data, max_data and group_data are needed when component_type is strategy
            if min_data:
                component._set_min_data(min_data)
            if max_data:
                component._set_max_data(max_data)
            component._set_group_data(group_data)
        else:
            is_remote = True
        component._add_consumer(self._proxy if is_remote else self)
        components[component.name] = component
        self.logger.debug(f"{self.name} added {component.name}")
    
        if self.is_remote() and not is_remote:
            # NOTE: returns None when adding a local component to a remote component to avoid returning a serialized (copied) object
            return None
        else:
            return component
    
    def add_model(
        self, 
        model: ModelT | ActorProxy,
        name: str='',
        min_data: int | None=None,
        max_data: int | None=None,
        group_data: bool=True,
        signal_cols: list[str] | None=None,
        ray_actor_options: dict | None=None,
        **ray_kwargs
    ) -> ModelT | ActorProxy | None:
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
        self, 
        feature: FeatureT | ActorProxy, 
        name: str='',
        min_data: int | None=None,
        max_data: int | None=None,
        group_data: bool=True,
        signal_cols: list[str] | None=None,
        ray_actor_options: dict | None=None,
        **ray_kwargs
    ) -> FeatureT | ActorProxy | None:
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
        self, 
        indicator: IndicatorT | ActorProxy,
        name: str='',
        min_data: int | None=None,
        max_data: int | None=None,
        group_data: bool=True,
        signal_cols: list[str] | None=None,
        ray_actor_options: dict | None=None,
        **ray_kwargs
    ) -> IndicatorT | ActorProxy | None:
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

    def _on_quote(self, data: QuoteData):
        product, bids, asks, ts = data.product, data.bids, data.asks, data.ts
        local_components = self._local_components.get(data, [])
        for component in local_components:
            component._on_quote(data)
            self._update_signals(data, component)
        self._append_to_df(data)
        self.on_quote(product, bids, asks, ts)

    def _on_tick(self, data: TickData):
        product, px, qty, ts = data.product, data.px, data.qty, data.ts
        for listener in self._listeners[data]:
            listener._on_tick(data)
            self._update_signals(data, listener)
        self._append_to_df(data)
        self.on_tick(product, px, qty, ts)
    
    def _on_bar(self, data: BarData):
        product, bar, ts = data.product, data.bar, data.bar.end_ts
        # TODO: wait for remote components' outputs
        for listener in self._listeners[data]:
            listener._on_bar(data)
            self._update_signals(data, listener)
        # TODO: non-wasm: send_signal, wasm: loop consumers.on_signal
        # TODO: add to trading store, self.store
        self._append_to_df(data)
        self.on_bar(product, bar, ts)

    def _update_signals(self, data: BaseData, listener: BaseStrategy | BaseModel):
        pred_y: torch.Tensor | np.ndarray | None = listener._next(data)
        if pred_y is not None:
            signal_cols = listener.get_signal_cols()
            for i, col in enumerate(signal_cols):
                self.predictions[col] = pred_y[i]
    
    def _gather(self):
        '''Sets up everything before start'''
        # NOTE: use is_gathered to avoid a component being gathered multiple times when it's a shared component
        if not self._is_gathered:
            self.add_datas()
            self.add_models()
            self.add_features()
            self.add_indicators()
            if self.env != Environment.BACKTEST:
                self.databoy._subscribe()
            self._is_gathered = True
            # TODO:
            # self._add_datas_from_consumer_if_none()
            
            assert self.store is not None, "trading store must be initialized before gathering"
            self.store.materialize()
            
            for component in self.components:
                component._gather()
            self.logger.info(f"'{self.name}' has gathered")
        else:
            self.logger.info(f"'{self.name}' has already gathered")
    
    def start(self):
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

    def add_datas(self):
        pass
    
    def add_models(self):
        pass

    def add_features(self):
        pass
    
    def add_indicators(self):
        pass
    
    def on_start(self):
        pass
    
    def on_stop(self):
        pass
    
    
    '''
    ************************************************
    Sugar Functions
    ************************************************
    '''
    def get_second_bar(self, product: BaseProduct, period: int) -> BarData | None:
        return self.get_data(product, resolution=f'{period}s')
    
    def get_minute_bar(self, product: BaseProduct, period: int) -> BarData | None:
        return self.get_data(product, resolution=f'{period}m')
    
    def get_hour_bar(self, product: BaseProduct, period: int) -> BarData | None:
        return self.get_data(product, resolution=f'{period}h')
    
    def get_day_bar(self, product: BaseProduct, period: int) -> BarData | None:
        return self.get_data(product, resolution=f'{period}d')
    