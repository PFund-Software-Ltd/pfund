# pyright: reportUninitializedInstanceVariable=false, reportUnknownParameterType=false, reportUnknownMemberType=false, reportArgumentType=false, reportAssignmentType=false, reportReturnType=false, reportAttributeAccessIssue=false, reportUnknownVariableType=false, reportOptionalMemberAccess=false, reportUnknownArgumentType=false
from __future__ import annotations
from typing import TYPE_CHECKING, Any, cast, Literal, ClassVar
if TYPE_CHECKING:
    from sklearn.base import BaseEstimator
    import torch.nn as nn
    from narwhals._native import NativeDataFrame
    from pfund.typing import (
        ComponentT,
        ModelT,
        IndicatorT,
        FeatureT,
        Component,
        ComponentName,
        ProductName,
        ColumnName,
    )
    from pfund.datas.stores.base_data_store import BaseDataStore
    from pfund.engines.engine_context import EngineContext
    from pfund.datas.timeframe import Timeframe
    from pfund.datas.data_market import MarketData
    from pfund.datas.data_base import BaseData
    from pfund.datas.stores.market_data_store import MarketDataStore
    from pfund.entities.products.product_base import BaseProduct
    from pfund.datas import QuoteData, TickData, BarData
    from pfund.engines.settings.trade_engine_settings import TradeEngineSettings
    from pfund.components.models.model_base import BaseModel
    from pfund.components.features.feature_base import BaseFeature
    from pfund.components.indicators.indicator_base import BaseIndicator
    from pfund.brokers.broker_base import BaseBroker
    
import logging
import datetime

import narwhals as nw

from pfund_kit.utils import toml
from pfund_kit.style import cprint, RichColor, TextStyle
from pfeed.enums import DataCategory
from pfeed.storages.storage_config import StorageConfig
from pfund.datas.resolution import Resolution
from pfund.datas.data_config import DataConfig
from pfund.components.actor_proxy import ActorProxy
from pfund.enums import ComponentType, RunMode, Environment, Broker, TradingVenue
from pfund.utils.decorators import ray_method
from pfund.datas.stores.trading_store import TradingStore
from pfund.datas.databoy import DataBoy


class ComponentMixin:
    config: ClassVar[dict[str, Any]] = {
        'max_rows': None,  # None means no limit, component will keep all data in memory
        'warmup_period': None,  # None means no warmup period, component will start right away
        'lookback_period': None,  # None means using the whole dataset
    }
    params: ClassVar[dict[str, Any]] = {}

    # custom post init for-common attributes of strategy and model
    def __mixin_post_init__(self, *args: Any, **kwargs: Any):
        self.__pfund_args__ = args
        self.__pfund_kwargs__ = kwargs
        
        self._name = self._get_default_name()
        self._run_mode: RunMode | None = None
        self._resolution: Resolution | None = None
        self._context: EngineContext | None = None
        
        self.logger: logging.Logger = logging.getLogger('pfund')
        self.databoy = DataBoy(self)
        self.store = TradingStore(self)

        self._signal_cols: list[str] = []

        self.products: dict[ProductName, BaseProduct] = {}
        self.models: dict[str, BaseModel | ActorProxy[BaseModel]] = {}
        self.features: dict[str, BaseFeature | ActorProxy[BaseFeature]] = {}
        self.indicators: dict[str, BaseIndicator | ActorProxy[BaseIndicator]] = {}

        self._is_running = False
        self._is_gathered = False
        self._is_top_component = False
        self._assert_functions_signatures()
    
    @property
    def env(self) -> Environment:
        assert self._context is not None, 'context is not set'
        return self._context.env
    
    @property
    def name(self) -> ComponentName:
        return self._name
    
    @property
    def run_mode(self) -> RunMode:
        assert self._run_mode is not None, 'run_mode is not set'
        return self._run_mode
    
    @property
    def context(self) -> EngineContext:
        assert self._context is not None, 'context is not set'
        return self._context
    
    @property
    def settings(self) -> TradeEngineSettings:
        return cast("TradeEngineSettings", self.context.settings)
    
    @property
    def market_data_store(self) -> MarketDataStore:
        return self.databoy.get_data_store(DataCategory.MARKET_DATA)
    
    @property
    def data_stores(self) -> dict[DataCategory, BaseDataStore]:
        return self.databoy._data_stores
    
    @property
    def storage_config(self) -> StorageConfig:
        return self.store._storage_config
    
    # TODO: also check on_bar, on_tick, on_quote etc.
    def _assert_functions_signatures(self):
        pass

    # useful when user wants to set logger specific to the component. currently 'pfund' is the default logger.
    def set_logger(self, logger: logging.Logger):
        self.logger = logger
        if self.is_remote(direct_only=False):
            self._setup_logging()
    
    def is_top_component(self) -> bool:
        return self._is_top_component

    def _hydrate(
        self,
        name: ComponentName,
        run_mode: RunMode,
        resolution: Resolution | str,
        engine_context: EngineContext,
        storage_config: StorageConfig | None=None,
        is_top_component: bool=False,
    ):
        """
        Hydrates the component with necessary attributes after initialization.
        
        Args:
            name (ComponentName): The name to assign to this component.
            run_mode (RunMode): The mode in which the component will run (e.g., local or remote).
            resolution (Resolution | str): The data resolution used by this component.
            engine_context (EngineContext): The engine context associated with this component. It is None if the component is running in a remote process.
            storage_config (StorageConfig): The storage configuration associated with this component.
        """
        self._context = engine_context
        self._run_mode = run_mode
        self._set_name(name)
        self._set_resolution(resolution)
        self.set_logger(self.logger)
        if storage_config is not None:
            self.store.set_storage_config(storage_config)
        self._is_top_component = is_top_component
        
    def _setup_messaging(self):
        self.databoy._setup_messaging()
        for component in self.components:
            component._setup_messaging()
        self.databoy._subscribe()
    
    def _setup_logging(self):
        '''Sets up logging for component running in remote process, uses zmq's PUSHHandler to send logs to engine'''
        from pfund.utils.zmq_pub_handler import ZMQPubHandler
        from pfeed.streaming.zeromq import ZeroMQ
        from pfund_kit.logging.configurator import LoggingDictConfigurator
        from pfund_kit.utils import get_free_port

        # configure logging based on pfund's logging config, e.g. log_level, log_file, log_format, etc.
        logging_configurator = LoggingDictConfigurator.create(
            log_path=self.context.pfund_config.log_path / self.env / self.context.name, 
            logging_config=self.context.logging_config, 
            lazy=True,
            use_colored_logger=True,
        )
        logging_configurator.configure()
        
        # remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
            handler.close()
            
        # add zmq PushHandler
        zmq_url = self.settings.zmq_urls.get(self.name, ZeroMQ.DEFAULT_URL)
        zmq_port = get_free_port()
        zmq_handler = ZMQPubHandler(f'{zmq_url}:{zmq_port}')
        zmq_formatter = logging.Formatter(
            fmt='%(message)s | from:%(filename)s fn:%(funcName)s ln:%(lineno)d (sent@%(asctime)s.%(msecs)03d)',
            datefmt='%H:%M:%S'
        )
        zmq_handler.setFormatter(zmq_formatter)
        self.logger.addHandler(zmq_handler)
        self.settings.zmq_urls.update({
            self.name: zmq_url 
        })
        self.settings.zmq_ports.update({
            self.name + '_logger': zmq_port
        })
    
    @classmethod
    def load_config(cls, config: dict[str, Any] | None=None, file_path: str=''):
        '''Loads config from a dict or a TOML file, overriding the class-level defaults.
        Args:
            config: Config dict to set directly.
            file_path: Path to a TOML file to load config from.
        '''
        if config and file_path:
            raise ValueError("config and file_path cannot be provided at the same time")
        if config:
            cls.config = config
        elif file_path:
            config = toml.load(file_path)
            if config is None:
                raise ValueError(f"Failed to load config from {file_path}")
            cls.config = config
    
    @classmethod
    def load_params(cls, params: dict[str, Any] | None=None, file_path: str=''):
        '''Loads params from a dict or a TOML file, overriding the class-level defaults.
        Args:
            params: Params dict to set directly.
            file_path: Path to a TOML file to load params from.
        '''
        if params and file_path:
            raise ValueError("params and file_path cannot be provided at the same time")
        if params:
            cls.params = params
        elif file_path:
            params = toml.load(file_path)
            if params is None:
                raise ValueError(f"Failed to load params from {file_path}")
            cls.params = params
    
    def _check_before_start(self):
        # check data
        if not self.get_datas():
            raise ValueError(f"{self.name} has no market data, did you forget to call add_datas()?")
        # check config
        if 'max_rows' not in self.config:
            raise ValueError(f'max_rows is not set in {self.name} config')
        if 'lookback_period' not in self.config:
            raise ValueError(f'lookback_period is not set in {self.name} config')
        if 'warmup_period' not in self.config:
            raise ValueError(f'warmup_period is not set in {self.name} config')
        max_rows = self.config['max_rows']
        warmup_period = self.config['warmup_period']
        lookback_period = self.config['lookback_period']
        if max_rows is None:
            cprint(
                f"'max_rows' is None. i.e. {self.name} data will be UNBOUNDED",
                style=TextStyle.BOLD + RichColor.YELLOW,
            )
        if warmup_period is None:
            cprint(
                f"'warmup_period' is None. i.e. {self.name} will be ready to compute signals IMMEDIATELY",
                style=TextStyle.BOLD + RichColor.YELLOW,
            )
            warmup_period = 0
        assert warmup_period >= 0, f'{self.name} {warmup_period=} is less than 0, please set warmup_period >= 0'
        if lookback_period is None:
            cprint(
                f"'lookback_period' is None. i.e. {self.name} will use the WHOLE DATASET to compute signals",
                style=TextStyle.BOLD + RichColor.YELLOW,
            )
        else:
            assert lookback_period >= 1, f'{self.name} {lookback_period=} is less than 1, please set lookback_period >= 1'
            if lookback_period > warmup_period:
                raise ValueError(f'{self.name} config: {lookback_period=} is greater than {warmup_period=}, please set lookback_period <= warmup_period')
    
    def to_dict(self) -> dict[str, Any]:
        return {
            'class_name': self.__class__.__name__,
            'signature': (self.__pfund_args__, self.__pfund_kwargs__),
            'env': self.env.value,
            'run_mode': self.run_mode.value,
            'name': self.name,
            'data_range': (self.context.data_start, self.context.data_end),
            'resolution': repr(self.resolution),
            'df_form': self._df_form,
            'component_type': self.component_type.value,
            'signal_cols': self._signal_cols,
            'config': self.config,
            'params': self.params,
            'settings': self.settings.model_dump(),
            'datas': [data.to_dict() for data in self.get_datas()],
            'models': [model.to_dict() for model in self.models.values()],
            'features': [feature.to_dict() for feature in self.features.values()],
            'indicators': [indicator.to_dict() for indicator in self.indicators.values()],
        }
    
    @property
    def components(self) -> list[Component | ActorProxy[Component]]:
        return self.get_components()
    
    def get_component(self, name: ComponentName) -> Component | ActorProxy[Component] | None:
        return self.models.get(name, None) or self.features.get(name, None) or self.indicators.get(name, None)
    
    def get_components(self) -> list[Component | ActorProxy[Component]]:
        return [
            *self.models.values(), 
            *self.features.values(), 
            *self.indicators.values(),
        ]
    
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
        else:
            raise ValueError(f"Unknown component type: {self.__class__.__name__}")
    
    @property
    def output_df(self) -> NativeDataFrame | None:
        return self.get_df(kind='output', window_size=None, to_native=True)
    
    @property
    def data_df(self) -> NativeDataFrame | None:
        return self.get_df(kind='data', data_category=None, window_size=None, to_native=True)
    
    @property
    def signals_df(self) -> NativeDataFrame | None:
        return self.get_df(kind='signals', window_size=None, to_native=True)
    
    @property
    def features_df(self) -> NativeDataFrame | None:
        return self.get_df(kind='features', window_size=None, to_native=True)
    df = features_df
    
    def get_df(
        self, 
        *,
        kind: Literal['data', 'signals', 'features', 'output'] = 'output',
        data_category: DataCategory | str | None=DataCategory.MARKET_DATA,
        window_size: int | None = None, 
        pivot_data: bool = False,
        to_native: bool = False
    ) -> nw.DataFrame[Any] | NativeDataFrame | None:
        """Returns one of the stored dataframes in either trading store or data stores.
    
        Args:
            kind: Which frame to return.
                - 'output': output dataframe generated by this component.
                - 'data': input dataframe from a data store (e.g. market data, news).
                - 'signals': signals dataframe used by this component.
                - 'features': features_df = data_df (merged data dfs from different categories) + signals_df (signals from other components)
            data_category: For kind='data', which data category to return. 
                If None, return a merged data_df from all data categories.
                Ignored when kind != 'data'. 
                Defaults to market data.
            window_size: Number of most recent rows to return.
                Defaults to None, i.e. return the whole dataframe.
            pivot_data: pivot data dataframe from long form to wide form. 
                Ignored when kind != 'data'. 
                Defaults to False.
            to_native: If True, return the underlying backend frame (polars/pandas) instead
                of a Narwhals DataFrame. Defaults to True.
        """
        def _get_data_dfs() -> dict[DataCategory, NativeDataFrame | None]:
            data_dfs: dict[DataCategory, NativeDataFrame | None] = {
                category: store.get_df(window_size=window_size, to_native=True)
                for category, store in self.data_stores.items()
            }
            if not all([df is not None for df in data_dfs.values()]):
                raise ValueError(f"Some data dfs are None for {self.name}, i.e. data are not ready, cannot get data_df")
            return data_dfs
        def _get_signals_dfs() -> dict[ComponentName, NativeDataFrame | None]:
            signals_dfs: dict[ComponentName, NativeDataFrame | None] = {
                component.name: component.get_trading_store().get_df(window_size=window_size, to_native=True)
                for component in self.components
            }
            if not all([df is not None for df in signals_dfs.values()]):
                raise ValueError(f"Some signals dfs are None for {self.name}, i.e. signals are not ready, cannot get signals_df")
            return signals_dfs
        
        if kind == 'output':
            df = self.store.get_df(window_size=window_size, to_native=True)
        elif kind == 'data':
            if data_category is not None:
                store = self.databoy.get_data_store(data_category)
                df = cast("NativeDataFrame | None", store.get_df(window_size=window_size, pivot=pivot_data, to_native=True))
            else:
                data_dfs = _get_data_dfs()
                df = self.merge_data_dfs(data_dfs)
        elif kind == 'signals':
            # NOTE: merging signals requires data_df to handle the (self._df_form='long', component_df_form='wide') case
            # e.g. strategy is in long form, its models are in wide form
            # using data_df (in long form=self._df_form) to help convert models signals_dfs in wide form to long form
            data_dfs = _get_data_dfs()
            data_df = self.merge_data_dfs(data_dfs)
            signals_dfs = _get_signals_dfs()
            df = self.merge_signals_dfs(data_df, signals_dfs) if signals_dfs else None
        elif kind == 'features':
            data_dfs = _get_data_dfs()
            data_df = self.merge_data_dfs(data_dfs)
            signals_dfs = _get_signals_dfs()
            signals_df = self.merge_signals_dfs(data_df, signals_dfs) if signals_dfs else None
            df = self.featurize(data_df, signals_df)
        else:
            raise ValueError(f"Invalid {kind=} for component {self.name}")
        if df is None:
            return None
        return df if to_native else nw.from_native(df)
    
    @ray_method
    def get_df_form(self) -> Literal['wide', 'long']:
        return self._df_form
    
    @ray_method
    def get_trading_store(self) -> TradingStore:
        return self.store
    
    def set_df_form(self, df_form: Literal['wide', 'long']):
        self._df_form = df_form.lower()
    
    def get_datas(self) -> list[BaseData]:
        datas = []
        for data_store in self.data_stores.values():
            datas.extend(data_store.get_datas())
        return datas
    
    @staticmethod
    def get_supported_resolutions(product: BaseProduct) -> dict[Timeframe, list[int]]:
        import importlib

        supported_resolutions: dict[Timeframe, list[int]]
        broker = product.broker
        if broker == Broker.CRYPTO:
            Exchange = getattr(importlib.import_module(f'pfund.brokers.crypto.exchanges.{product.exchange.lower()}.exchange'), 'Exchange')
            supported_resolutions = Exchange.get_supported_resolutions(product)
        elif broker == Broker.IBKR:
            InteractiveBrokersAPI = getattr(importlib.import_module('pfund.brokers.ibkr.api'), 'InteractiveBrokersAPI')
            supported_resolutions = InteractiveBrokersAPI.SUPPORTED_RESOLUTIONS
        else:
            raise NotImplementedError(f'broker {broker} is not supported')
        return supported_resolutions

    @staticmethod
    def dt(ts: float, tz: datetime.tzinfo = datetime.timezone.utc) -> datetime.datetime:
        from pfund_kit.utils.temporal import convert_ts_to_dt
        return convert_ts_to_dt(ts, tz=tz)
    
    @staticmethod
    def now(tz: datetime.tzinfo = datetime.timezone.utc) -> datetime.datetime:
        return datetime.datetime.now(tz=tz)
    
    @property
    def resolution(self) -> Resolution:
        assert self._resolution is not None, 'resolution is not set'
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
        self._name = name
        if not self._name.lower().endswith(self.component_type):
            self._name += f"_{self.component_type}"
        
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
                - If True (default), only checks the component's own `run_mode`.
                  Reflects whether the component *itself* is declared to be remote.
                  e.g. a model running inside a strategy (ray actor) is "local" relative to itself.
                - If False, also checks whether this component's code is currently
                  executing inside a Ray actor process via `ray.get_runtime_context()`.
                  This captures the case where a declaratively local component is
                  nested inside a remote parent and therefore runs in the parent's
                  actor process.

        Returns:
            bool: True if the component is declared remote, or (when `direct_only=False`)
                  is currently executing inside a Ray actor process.
        """
        assert self.run_mode is not None, f"{self.name} has no run mode"
        is_remote = self.run_mode == RunMode.REMOTE
        if is_remote or direct_only:
            return is_remote
        try:
            import ray
            if not ray.is_initialized():
                return False
            return ray.get_runtime_context().get_actor_id() is not None
        except Exception:
            return False
        return False
    
    def _add_product(
        self,
        venue: TradingVenue | str,
        basis: str,
        exch: str='',
        symbol: str='',
        name: str='',
        **specs: Any
    ) -> BaseProduct:
        from pfund.brokers import create_broker
        # NOTE: broker is only used to create product but nothing else
        broker: BaseBroker = create_broker(env=self.env, bkr=TradingVenue[venue.upper()].broker, settings=self.settings)
        if broker.name == Broker.CRYPTO:
            exch = venue
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
    
    def get_data(self, product: ProductName, resolution: Resolution | str) -> MarketData | None:
        return self.market_data_store.get_data(product, resolution)
    
    def _get_default_name(self):
        return self.__class__.__name__
    
    def add_data(
        self, 
        venue: TradingVenue | str,
        product: str,
        resolutions: list[Resolution | str] | None=None,
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
            resolutions: data resolutions in use, e.g. "1t" for tick data, "1q" for quote data
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
            venue=venue,
            basis=product,
            exch=exchange,
            symbol=symbol,
            name=product_name,
            **product_specs
        )
        datas: list[MarketData] = self.market_data_store.add_data(
            product=product, 
            resolutions=resolutions,
            data_config=data_config,
            storage_config=storage_config,
        )
        return datas
    
    def _add_component(
        self, 
        component: ComponentT | ActorProxy[ComponentT],
        resolution: str='',
        name: str='', 
        storage_config: StorageConfig | None=None,
        ray_actor_options: dict[str, Any] | None=None,
        **ray_kwargs: Any
    ) -> ComponentT | ActorProxy[ComponentT] | None:
        '''Adds a model component to the current component.
        A model component is a model, feature, or indicator.
        Args:
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
                component = ActorProxy(
                    component, 
                    name=component_name,
                    resolution=resolution or self.resolution,
                    component_type=component_type,
                    engine_context=self.context,
                    ray_actor_options=ray_actor_options, 
                    **ray_kwargs
                )
                    
            component._hydrate(
                name=component_name,
                run_mode=RunMode.REMOTE if ray_kwargs else RunMode.LOCAL,
                resolution=resolution or self.resolution,
                engine_context=self.context,
                storage_config=storage_config,
            )
        
        components[component.name] = component
        self.logger.debug(f"{self.name} added {component.name}")
    
        # NOTE: returns None when adding a local component (not ActorProxy) to a remote component to avoid returning a serialized (copied) object
        if self.is_remote() and not isinstance(component, ActorProxy):
            return None
        return component
    
    def add_model(
        self, 
        model: ModelT | ActorProxy[ModelT] | BaseEstimator | nn.Module,
        resolution: str='',
        name: str='',
        storage_config: StorageConfig | None=None,
        ray_actor_options: dict[str, Any] | None=None,
        **ray_kwargs: Any
    ) -> ModelT | ActorProxy[ModelT] | None:
        from pfund.components.models.model_base import wrap_model
        model = wrap_model(model)
        return self._add_component(
            component=model,
            resolution=resolution,
            name=name,
            storage_config=storage_config,
            ray_actor_options=ray_actor_options,
            **ray_kwargs
        )
    
    def add_feature(
        self, 
        feature: FeatureT | ActorProxy[FeatureT], 
        resolution: str='',
        name: str='',
        storage_config: StorageConfig | None=None,
        ray_actor_options: dict[str, Any] | None=None,
        **ray_kwargs: Any
    ) -> FeatureT | ActorProxy[FeatureT] | None:
        return self._add_component(
            component=feature, 
            resolution=resolution,
            name=name, 
            storage_config=storage_config,
            ray_actor_options=ray_actor_options,
            **ray_kwargs
        )
    
    def add_indicator(
        self, 
        indicator: IndicatorT | ActorProxy[IndicatorT],
        resolution: str='',
        name: str='',
        storage_config: StorageConfig | None=None,
        ray_actor_options: dict[str, Any] | None=None,
        **ray_kwargs: Any
    ) -> IndicatorT | ActorProxy[IndicatorT] | None:
        return self._add_component(
            component=indicator,
            resolution=resolution,
            name=name,
            storage_config=storage_config,
            ray_actor_options=ray_actor_options,
            **ray_kwargs
        )
    
    # TODO
    def _on_quote(self, data: QuoteData):
        self.on_quote(data)

    # TODO
    def _on_tick(self, data: TickData):
        self.on_tick(data)
    
    def _on_bar(self, data: BarData):
        if self.is_ready(data=data):
            lookback_period = self.config['lookback_period']
            self.run_pipeline(lookback_period=lookback_period)
        self.on_bar(data)
    
    def set_signal_cols(self, signal_cols: list[str]):
        self._signal_cols = signal_cols
    
    def set_pivot_cols(self, pivot_cols: list[str]):
        self.store.set_pivot_cols(pivot_cols)
    
    def is_ready(self, data: BaseData) -> bool:
        warmup_period = self.config['warmup_period']
        if data.category == DataCategory.MARKET_DATA:
            df = self.get_df(kind='data', data_category=data.category, to_native=False)
            if df is None or len(df) < warmup_period:
                return False
            if self._df_form == 'long':
                # long form: ready when this product's bar is closed
                return data.is_closed()
            elif self._df_form == 'wide':
                # wide form: ready when all products' bars are closed
                return all(
                    cast("BarData", self.get_data(product.name, self.resolution)).is_closed()
                    for product in self.products.values()
                )
        # EXTEND:
        else:
            raise NotImplementedError(f'is_ready() is not implemented for {data.category=}')
    
    def merge_data_dfs(self, data_dfs: dict[DataCategory, NativeDataFrame]) -> NativeDataFrame:
        '''Creates data_df by merging data_dfs per data category in long form
        Args:
            data_dfs: dataframes per data category in long form
        '''
        dfs: list[nw.DataFrame[Any]] = []
        common_key_cols: set[str] | None = None
        for category, df in data_dfs.items():
            data_store = self.databoy.get_data_store(category)
            nw_df = nw.from_native(df)
            # strip provenance metadata: the merged data_df feeds featurize, where metadata is not a feature
            metadata_to_drop = [col for col in data_store.METADATA_COLS if col in nw_df.columns]
            if metadata_to_drop:
                nw_df = nw_df.drop(metadata_to_drop)
            if self._df_form == 'wide':
                # REVIEW: this requires dynamic pivoting for EACH call
                nw_df = data_store.pivot_df(nw_df)
            dfs.append(nw_df)
            # KEY_COLS that actually survive on this df (pivot_df folds PIVOT_COLS into column names)
            df_key_cols = set(col for col in data_store.KEY_COLS if col in nw_df.columns)
            if common_key_cols is None:
                common_key_cols = df_key_cols
            else:
                common_key_cols &= df_key_cols

        data_df = dfs[0]
        if len(dfs) == 1:
            return data_df.to_native()
        
        if not common_key_cols:
            raise ValueError(
                f"No common key columns found in data_dfs in {self._df_form} form for {self.name}. " +
                "Please define your own merge_data_dfs() to handle merging data_dfs manually."
            )
            
        for df in dfs[1:]:
            data_df = data_df.join(df, on=list(common_key_cols), how='full')
            for key in common_key_cols:
                right_key = f'{key}_right'
                data_df = data_df.with_columns(nw.coalesce(key, right_key).alias(key)).drop(right_key)
        return data_df.sort(common_key_cols).to_native()
    
    def merge_signals_dfs(self, data_df: NativeDataFrame, signals_dfs: dict[ComponentName, NativeDataFrame]) -> NativeDataFrame:
        '''Creates signals_df by merging signals_dfs (signals from other components)
        Args:
            data_df: data_df in {self._df_form} form
            signals_dfs: dict of signals_df per component name
        Returns:
            signals_df: signals_df in {self._df_form} form
        '''
        dfs: list[nw.DataFrame[Any]] = []
        # (self._df_form='long', component_df_form='wide') case: 
        # cannot melt back to long, so merge them in afterwards by broadcasting on the index col only
        special_dfs: list[nw.DataFrame[Any]] = []
        common_index_col: str | None = None
        common_key_cols: set[str] | None = None
        for component_name, df in signals_dfs.items():
            component = self.get_component(component_name)
            component_df_form = component.get_df_form()
            trading_store = component.get_trading_store()
            if common_index_col is None:
                common_index_col = trading_store.INDEX_COL
            else:
                if common_index_col != trading_store.INDEX_COL:
                    raise ValueError(f"Unhandled case: index column {common_index_col} is not the same as {trading_store.INDEX_COL} for {component_name}")
            # special case, cannot unpivot 'wide' to 'long'
            if self._df_form == 'long' and component_df_form == 'wide':
                special_dfs.append(nw.from_native(df))
                continue
            elif self._df_form == 'wide' and component_df_form == 'long':
                # REVIEW: this requires dynamic pivoting for EACH call
                nw_df = trading_store.pivot_df(nw.from_native(df))
            else:
                nw_df = nw.from_native(df)
            dfs.append(nw_df)
            # KEY_COLS that actually survive on this df (pivot_df folds PIVOT_COLS into column names)
            df_key_cols = set(col for col in trading_store.KEY_COLS if col in nw_df.columns)
            if common_key_cols is None:
                common_key_cols = df_key_cols
            else:
                common_key_cols &= df_key_cols
        
        if not dfs and not special_dfs:
            raise ValueError(
                f"Invalid input: {signals_dfs=} in merge_signals_dfs for {self.name}: no dataframe to merge"
            )
            
        # only special_dfs exist (e.g. self is strategy in long form, components are models in wide form)
        if not dfs:
            # special_dfs exist => signals_df is in self._df_form='long' form
            signals_df = nw.from_native(data_df)  # in long form
            common_key_cols = set(col for col in self.store.KEY_COLS if col in signals_df.columns)
        else:
            signals_df = dfs[0]
            if len(dfs) > 1:
                if not common_key_cols:
                    raise ValueError(
                        f"No common key columns found in signals_dfs in {self._df_form} form for {self.name}. " +
                        "Please define your own merge_signals_dfs() to handle merging signals_dfs manually."
                    )
                for df in dfs[1:]:
                    signals_df = signals_df.join(df, on=list(common_key_cols), how='full')
                    for key in common_key_cols:
                        right_key = f'{key}_right'
                        signals_df = signals_df.with_columns(nw.coalesce(key, right_key).alias(key)).drop(right_key)
        
        # broadcast (self._df_form='long', component_df_form='wide') specials on index col only
        for df in special_dfs:
            signals_df = signals_df.join(df, on=common_index_col, how='full')
            right_key = f'{common_index_col}_right'
            signals_df = signals_df.with_columns(nw.coalesce(common_index_col, right_key).alias(common_index_col)).drop(right_key)
        
        sort_keys = list(common_key_cols) if common_key_cols else [common_index_col]
        return signals_df.sort(sort_keys).to_native()
    
    def featurize(self, data_df: NativeDataFrame, signals_df: NativeDataFrame | None) -> NativeDataFrame:
        '''Creates features_df = data_df + signals_df (combined signals from other components)
        In machine learning, features_df is the X in predict(X).
        Args:
            data_df: dataframe in {self._df_form} form
            signals_df: signals_df from other components
        '''
        # no signals to combine with, features_df = data_df
        if signals_df is None:
            return data_df
        data = nw.from_native(data_df)
        signals = nw.from_native(signals_df)
        join_cols = [col for col in data.columns if col in signals.columns]
        if not join_cols:
            raise ValueError(
                f"No common columns between {self.name}'s data_df {data.columns} and signals_df {signals.columns}"
            )
        features_df = data.join(signals, on=join_cols, how='left')
        return features_df.to_native()
    
    def run_pipeline(self, lookback_period: int | None = None):
        '''
        Args:
            lookback_period: Number of most recent rows to run the pipeline on.
                Defaults to None, i.e. run the pipeline on the whole dataset.
        '''
        # TODO: wait for components' signals, somehow pass in lookback_period to the child components?
        if self.databoy.is_using_zmq():
            pass
        #     # TODO: this should return each component's signals_df (window sized)
        #     # self.databoy._wait_for_children_signals()
        else:
            for component in self.components:
                component.run_pipeline(lookback_period=lookback_period)
        features_df = cast("NativeDataFrame", self.get_df(kind='features', window_size=lookback_period, to_native=True))
        signals = cast("dict[ColumnName, Any]", self.signalize(features_df))
        self.store.update_df(features_df, signals)
        
    def _materialize(self):
        for data_store in self.data_stores.values():
            data_store.materialize()
        self.store.materialize()
        
    def _gather(self):
        '''Sets up everything before start'''
        # NOTE: use is_gathered to avoid a component being gathered multiple times when it's a shared component
        if not self._is_gathered:
            self._check_before_start()
            self.add_datas()
            self.add_models()
            self.add_features()
            self.add_indicators()
            for component in self.components:
                component._gather()
            self._materialize()
            self._is_gathered = True
            self.logger.info(f"'{self.name}' has gathered")
        else:
            self.logger.info(f"'{self.name}' has already gathered")
    
    def start(self):
        if not self.is_running():
            # set the ZMQPubHandler's receiver ready to flush the buffered log messages
            if self.is_remote(direct_only=False):
                self.logger.handlers[0].set_receiver_ready()
            for component in self.components:
                component.start()
            self._is_running = True
            self.on_start()
            self.databoy.start()
            self.logger.info(f"'{self.name}' has started")
        else:
            self.logger.info(f"'{self.name}' has already started")
    
    def stop(self, reason: str=''):
        '''Stops the component, keep the internal states'''
        if self.is_running():
            for component in self.components:
                component.stop()
            self._is_running = False
            self.on_stop()
            self.databoy.stop()
            self.logger.info(f"'{self.name}' has stopped, ({reason=})")
        else:
            self.logger.info(f"'{self.name}' has already stopped")

    
    '''
    ************************************************
    Override Methods
    Override these methods in your subclass to implement your custom behavior.
    ************************************************
    '''
    def on_quote(self, data: QuoteData):
        raise NotImplementedError(f"Please define your own on_quote(data: QuoteData) in your strategy '{self.name}'.")
    
    def on_tick(self, data: TickData):
        raise NotImplementedError(f"Please define your own on_tick(data: TickData) in your strategy '{self.name}'.")

    def on_bar(self, data: BarData):
        raise NotImplementedError(f"Please define your own on_bar(data: BarData) in your strategy '{self.name}'.")

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
    def get_orderbook(self, product: ProductName) -> MarketData | None:
        return self.get_data(product, Resolution('1q'))
    
    def get_tradebook(self, product: ProductName) -> MarketData | None:
        return self.get_data(product, Resolution('1t'))
