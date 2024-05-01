# NOTE: need this to make TYPE_CHECKING work to avoid the circular import issue
from __future__ import annotations

import os
import sys
import importlib
from pathlib import Path
from abc import ABC, abstractmethod
from collections import defaultdict

from typing import TYPE_CHECKING, Any, Union

try:
    import joblib
    import torch
    import torch.nn as nn
    from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
    from sklearn.pipeline import Pipeline
    import numpy as np
    import pandas as pd
    import polars as pl
except ImportError:
    pass

if TYPE_CHECKING:
    from pfund.strategies.strategy_base import BaseStrategy
    from pfund.models import PytorchModel, SklearnModel
    from pfund.indicators.indicator_base import TaFunction, TalibFunction
    from pfund.datas.data_base import BaseData
    from pfund.datas.data_quote import QuoteData
    from pfund.datas.data_tick import TickData
    from pfund.datas.data_bar import BarData
    from pfund.types.core import tModel, tFeature, tIndicator
    MachineLearningModel = Union[
        nn.Module,
        BaseEstimator,
        ClassifierMixin,
        RegressorMixin, 
        Pipeline,
        TaFunction,  # ta.utils.IndicatorMixin
        TalibFunction,
        Any,
    ]

from pfund.datas.resolution import Resolution
from pfund.models.model_meta import MetaModel
from pfund.products.product_base import BaseProduct
from pfund.utils.utils import short_path, get_engine_class, load_yaml_file, convert_ts_to_dt
from pfund.plogging import create_dynamic_logger


class BaseModel(ABC, metaclass=MetaModel):
    
    _file_path: Path | None = None  # Get the file path where the model was defined
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
    
    def load_params(self, params: dict | None=None):
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
            if strategy_file_path := getattr(module, '__file__', None):
                cls._file_path = Path(strategy_file_path)
                cls.load_config()
        return super().__new__(cls)
    
    def __init__(self, ml_model: MachineLearningModel, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
        self.name = self.mdl = self.__class__.__name__
        self.Engine = get_engine_class()
        self.engine = self.Engine()
        data_tool: str = self.Engine.data_tool
        DataTool = getattr(importlib.import_module(f'pfund.data_tools.data_tool_{data_tool}'), f'{data_tool.capitalize()}DataTool')
        self._data_tool = DataTool()
        self.logger = None
        self._is_running = False
        # minimum number of data required for the model to make a prediction
        self.min_data_points = kwargs.get('min_data_points', 1)
        self.max_data_points = kwargs.get('max_data_points', self.min_data_points)
        assert self.max_data_points >= self.min_data_points, f'max_data_points={self.max_data_points} must be greater than or equal to min_data_points={self.min_data_points}'
        self.ml_model = ml_model  # user-defined machine learning model
        self.type = 'model'
        self.products = defaultdict(dict)  # {trading_venue: {pdt1: product1, pdt2: product2} }
        self.datas = defaultdict(dict)  # {product: {'1m': data}}
        self._listeners = defaultdict(list)  # {data: model}
        self._consumers = []  # strategies/models that consume this model
        self.models = {}
        self.predictions = {}
        self.data = None  # last data
        self.signal = None  # output signal df from trained ml_model
        
        self.params = {}
        self.load_params()

    @abstractmethod
    def predict(self, *args, **kwargs) -> pd.DataFrame | pl.DataFrame | pl.LazyFrame | torch.Tensor | np.ndarray:
        pass
    
    def __getattr__(self, attr):
        '''gets triggered only when the attribute is not found'''
        if 'ml_model' in self.__dict__ and hasattr(self.ml_model, attr):
            return getattr(self.ml_model, attr)
        else:
            class_name = self.__class__.__name__
            raise AttributeError(f"'{class_name}' object or '{class_name}.ml_model' or '{class_name}.data_tool' has no attribute '{attr}'")
    
    @property
    def tname(self):
        '''type + name, e.g. model XYZ'''
        return f"{self.type} '{self.name}'"
    
    @property
    def df(self):
        return self._data_tool.df
    
    @property
    def index(self):
        return self._data_tool.index
    
    @property
    def group(self):
        return self._data_tool.group
    
    def get_df(self, copy=True):
        return self._data_tool.get_df(copy=copy)
    
    def get_data_tool(self):
        return self._data_tool
    
    @staticmethod
    def dt(ts: float):
        return convert_ts_to_dt(ts)
    
    def to_dict(self):
        return {
            'class': self.__class__.__name__,
            'name': self.name,
            'config': self.config,
            'params': self.params,
            'ml_model': self.ml_model,
            'datas': [repr(data) for product in self.datas for data in self.datas[product].values()],
            'models': [model.to_dict() for model in self.models.values()],
        }
    
    def output_df_to_parquet(self, file_path: str):
        self._data_tool.output_df_to_parquet(self.df, file_path)
    
    # if not specified, features are just the original df
    def prepare_features(self) -> pd.DataFrame | pl.DataFrame | pl.LazyFrame:
        return self.df
    
    def set_signal(self, signal: pd.DataFrame | pl.DataFrame | pl.LazyFrame | None):
        self.signal = signal
        
    def to_signal(self, X: pd.DataFrame | pl.DataFrame | pl.LazyFrame, pred_y: torch.Tensor | np.ndarray, columns: list[str] | None=None) -> pd.DataFrame | pl.DataFrame | pl.LazyFrame:
        if type(pred_y) is torch.Tensor:
            pred_y = pred_y.detach().numpy() if pred_y.requires_grad else pred_y.numpy()
        if not columns:
            num_cols = pred_y.shape[-1]
            if num_cols == 1:
                columns = [self.name]
            else:
                columns = [f'{self.name}_{i}' for i in range(num_cols)]
        dtl = self.get_data_tool()
        signal = dtl.to_signal(X, pred_y, columns=columns)
        self.set_signal(signal)
        return signal
    
    # FIXME: pandas specific
    def append_signal(self, X: pd.DataFrame | pl.DataFrame | pl.LazyFrame, new_pred: torch.Tensor | np.ndarray) -> pd.DataFrame | pl.DataFrame | pl.LazyFrame:
        '''Appends new signal to self.signal'''
        # self.data is the lastest data passed in
        index_data = {'ts': self.data.dt, 'product': repr(self.data.product), 'resolution': repr(self.data.resolution)}
        index = self._data_tool.create_multi_index(index_data, X.index.names)
        new_pred = new_pred.reshape(1, -1)
        signal = pd.DataFrame(new_pred, index=index, columns=self.signal.columns)
        signal = pd.concat([self.signal, signal], ignore_index=False)
        self.set_signal(signal)
        return signal
    
    def flow(self, is_dump=True) -> pd.DataFrame | pl.DataFrame | pl.LazyFrame:
        X: pd.DataFrame = self.prepare_features()
        pred_y: np.ndarray = self.predict(X)
        # No training
        signal: pd.DataFrame = self.to_signal(X, pred_y)
        if is_dump:
            self.dump(signal)
        return signal
    
    # FIXME: pandas specific
    def next(self) -> torch.Tensor | np.ndarray | None:
        '''Returns the next prediction in event-driven mode.'''
        # TODO: check if the pred_y is already there first
        # since the same data could be passed in multiple times 
        # if the model is a listener to multiple consumers (e.g. strategy+model)
        
        # get the lastest df (features) using self.min_data_points
        mask = self.df.index.get_level_values('ts') <= self.data.dt  # NOTE: self.data is the latest data
        positions = [i for i, m in enumerate(mask) if m]
        start_idx = positions[-max(self.min_data_points, self.max_data_points)] if len(positions) >= self.min_data_points else 0
        end_idx = positions[-1] if positions else -1  # -1 if positions is empty
        X = self.df[ start_idx : end_idx+1 ]
        
        # predict
        pred_y = self.predict(X)
        if pred_y is None:
            return
        
        # check if predictions are all nans
        num_rows = pred_y.shape[0]
        is_enough_data = num_rows >= self.min_data_points
        new_pred = pred_y[-1]
        if torch.is_tensor(new_pred):
            is_all_nan = torch.isnan(new_pred).all()
        elif isinstance(new_pred, np.ndarray):
            is_all_nan = np.all(np.isnan(new_pred))
        else:
            raise Exception(f'Unexpected new_pred type {type(new_pred)}')
        if is_enough_data and is_all_nan:
            raise Exception(f'wrong min_data_points={self.min_data_points} for model "{self.name}", got all nans predictions, try to increase your min_data_points')
        
        # initialize signal
        if self.signal is None:
            self.to_signal(X, pred_y)
        # update signal
        else:
            self.append_signal(X, new_pred)
        return new_pred
            
    def get_model_type_of_ml_model(self) -> PytorchModel | SklearnModel | BaseModel:
        from pfund.models import PytorchModel, SklearnModel
        if isinstance(self.ml_model, nn.Module):
            Model = PytorchModel
        elif isinstance(self.ml_model, (BaseEstimator, ClassifierMixin, RegressorMixin, Pipeline)):
            Model = SklearnModel
        else:
            Model = BaseModel
        return Model
    
    def add_consumer(self, consumer: BaseStrategy | BaseModel):
        '''
        when a model is added to a strategy, consumer is a strategy
        when a model is added to a model, consumer is a model
        '''
        if consumer not in self._consumers:
            self._consumers.append(consumer)

    def create_logger(self):
        if self.is_indicator():
            type_ = 'indicator'
        elif self.is_feature():
            type_ = 'feature'
        else:
            type_ = 'model'
        self.logger = create_dynamic_logger(self.name, type_)
        
    def set_name(self, name: str):
        self.name = self.mdl = name
        
    def _get_file_path(self, extension='.joblib'):
        path = f'{self.engine.config.artifact_path}/{self.name}'
        file_name = f'{self.name}{extension}'
        if not os.path.exists(path):
            os.makedirs(path)
        return f"{path}/{file_name}"
    
    def _assert_no_missing_datas(self, obj):
        loaded_datas = {data for product in obj['datas'] for data in obj['datas'][product].values()}
        added_datas = {data for product in self.datas for data in self.datas[product].values()}
        if loaded_datas != added_datas:
            missing_datas = loaded_datas - added_datas
            raise Exception(f"missing data {missing_datas} in model '{self.name}', please use add_data() to add them back")
    
    def load(self):
        file_path = self._get_file_path()
        if os.path.exists(file_path):
            obj = joblib.load(file_path)
            signal = obj['signal']
            if not signal:
                self.logger.warning(f'failed to load signal, please make sure model {self.name} was dumped using "model.dump(signal)" correctly.')
            self.set_signal(signal)
            self.ml_model = obj['ml_model']
            self._assert_no_missing_datas(obj)
            self.logger.debug(f"loaded trained ml_model '{self.name}' and its signal from {short_path(file_path)}")
        else:
            self.logger.debug(f"no trained ml_model '{self.name}' found in {short_path(file_path)}")
    
    def dump(self, signal: pd.DataFrame | pl.DataFrame | pl.LazyFrame):
        obj = {
            'signal': signal,
            'ml_model': self.ml_model,
            'datas': self.datas,
            # TODO: dump dates as well
        }
        file_path = self._get_file_path()
        joblib.dump(obj, file_path, compress=True)
        self.logger.debug(f"dumped trained ml_model '{self.name}' and its signal to {short_path(file_path)}")
    
    def add_listener(self, listener: BaseModel, listener_key: BaseData):
        if listener not in self._listeners[listener_key]:
            self._listeners[listener_key].append(listener)
    
    def remove_listener(self, listener: BaseModel, listener_key: BaseData):
        if listener in self._listeners[listener_key]:
            self._listeners[listener_key].remove(listener)
    
    # TODO
    def is_ready(self):
        pass
    
    def is_running(self):
        return self._is_running
    
    def is_indicator(self) -> bool:
        from pfund.indicators.indicator_base import BaseIndicator
        return isinstance(self, BaseIndicator)
    
    def is_feature(self) -> bool:
        return isinstance(self, BaseFeature)
    
    def _is_signal_prepared(self):
        return True
    
    def get_datas(self) -> list[BaseData]:
        datas = []
        for product in self.datas:
            datas.extend(list(self.datas[product].values()))
        return datas
    
    def set_data(self, product: BaseProduct, resolution: str | Resolution, data: BaseData):
        if isinstance(resolution, Resolution):
            resolution = repr(resolution)
        self.datas[product][resolution] = data

    def get_data(self, product: BaseProduct, resolution: str | None=None):
        return self.datas[product] if not resolution else self.datas[product][resolution]
    
    def add_data(self, trading_venue, base_currency, quote_currency, ptype, *args, **kwargs) -> list[BaseData]:
        datas = []
        for consumer in self._consumers:
            consumer_datas = self._add_consumer_datas(consumer, trading_venue, base_currency, quote_currency, ptype, *args, **kwargs)
            datas += consumer_datas
        return datas
    
    def _add_consumer_datas(self, consumer: BaseStrategy | BaseModel, *args, use_consumer_data=False, **kwargs) -> list[BaseData]:
        if not use_consumer_data:
            consumer_datas = consumer.add_data(*args, **kwargs)
        else:
            consumer_datas = consumer.get_datas()
        for data in consumer_datas:
            self.set_data(data.product, data.resolution, data)
            consumer.add_listener(listener=self, listener_key=data)
        return consumer_datas
    
    # IMPROVE: current issue is in next(), when the df has multiple products and resolutions,
    # don't know how to determine the exact minimum amount of data points for predict()
    def _adjust_min_data_points(self):
        num_products = self.df['product'].nunique()
        num_resolutions = self.df['resolution'].nunique()
        assert num_products and num_resolutions, f"{num_products=} and/or {num_resolutions=} are invalid, please check your dataframe"
        adj_min_data_points = self.min_data_points * num_products * num_resolutions
        adj_max_data_points = self.max_data_points * num_products * num_resolutions
        self.logger.warning(f'adjust min_data_points from {self.min_data_points} to {adj_min_data_points} with {num_products=} and {num_resolutions=}')
        self.logger.warning(f'adjust max_data_points from {self.max_data_points} to {adj_max_data_points} with {num_products=} and {num_resolutions=}')
        self.min_data_points = adj_min_data_points
        self.max_data_points = adj_max_data_points
        
    def get_model(self, name: str) -> BaseModel:
        return self.models[name]
    
    def add_model(self, model: tModel, name: str='') -> tModel:
        Model = model.get_model_type_of_ml_model()
        assert isinstance(model, Model), \
            f"{model.type} '{model.__class__.__name__}' is not an instance of {Model.__name__}. Please create your {model.type} using 'class {model.__class__.__name__}({Model.__name__})'"
        if name:
            model.set_name(name)
        model.create_logger()
        mdl = model.name
        if mdl in self.models:
            raise Exception(f"{model.tname} already exists in {self.tname}")
        model.add_consumer(self)
        self.models[mdl] = model
        self.logger.debug(f"added {model.tname}")
        return model
    
    def add_feature(self, feature: tFeature, name: str='') -> tFeature:
        return self.add_model(feature, name=name)
    
    def add_indicator(self, indicator: tIndicator, name: str='') -> tIndicator:
        return self.add_model(indicator, name=name)
    
    def update_quote(self, data: QuoteData, **kwargs):
        product, bids, asks, ts = data.product, data.bids, data.asks, data.ts
        self.data = data
        for listener in self._listeners[data]:
            listener.update_quote(data, **kwargs)
            self.update_predictions(listener)
        self._append_to_df(**kwargs)
        self.on_quote(product, bids, asks, ts, **kwargs)
        
    def update_tick(self, data: TickData, **kwargs):
        product, px, qty, ts = data.product, data.px, data.qty, data.ts
        self.data = data
        for listener in self._listeners[data]:
            listener.update_tick(data, **kwargs)
            self.update_predictions(listener)
        self._append_to_df(**kwargs)
        self.on_tick(product, px, qty, ts, **kwargs)
    
    def update_bar(self, data: BarData, **kwargs):
        product, bar, ts = data.product, data.bar, data.bar.end_ts
        self.data = data
        for listener in self._listeners[data]:
            listener.update_bar(data, **kwargs)
            self.update_predictions(listener)
        self._append_to_df(**kwargs)
        self.on_bar(product, bar, ts, **kwargs)
    
    def update_predictions(self, listener: BaseModel):
        pred_y: torch.Tensor | np.ndarray | None = listener.next()
        self.predictions[listener.name] = pred_y
    
    def _start_models(self):
        for model in self.models.values():
            model.start()

    def _prepare_df(self):
        return self._data_tool.prepare_df()
        
    def _append_to_df(self, **kwargs):
        return self._data_tool.append_to_df(self.data, self.predictions, **kwargs)
    
    def start(self):
        if not self.is_running():
            self.add_datas()
            if not self.datas:
                self.logger.warning(f"No data for {self.tname}, adding datas from consumers {[consumer.tname for consumer in self._consumers]}")
                for consumer in self._consumers:
                    self._add_consumer_datas(consumer, use_consumer_data=True)
            self.add_models()
            self._start_models()
            self._prepare_df()
            self.load()  # load trained model, set signal
            # prepare indicator's signal on the fly if required
            if self._is_signal_prepared() and self.signal is None:
                if self.is_indicator():
                    self.logger.debug(f'calculating indicator {self.name} signal(s) on the fly')
                    self.flow(is_dump=False)
                else:
                    raise Exception(f"signal is None, please make sure model '{self.name}' is loaded or was dumped using 'model.dump(signal)' correctly.")
            self._adjust_min_data_points()
            self.on_start()
            self._is_running = True
        else:
            self.logger.warning(f'model {self.name} has already started')
        
    def stop(self):
        if self.is_running():
            self._is_running = False
            self.on_stop()
            for model in self.models.values():
                model.stop()
        else:
            self.logger.warning(f'strategy {self.name} has already stopped')
        
    '''
    ************************************************
    Model Functions
    Users can customize these functions in their models.
    ************************************************
    '''
    def add_datas(self):
        pass
    
    def add_models(self):
        pass
    
    def on_start(self):
        pass
    
    def on_stop(self):
        pass
    
    def on_quote(self, product, bids, asks, ts, **kwargs):
        pass
    
    def on_tick(self, product, px, qty, ts, **kwargs):
        pass

    def on_bar(self, product, bar, ts, **kwargs):
        pass
    
    
class BaseFeature(BaseModel):
    '''Feature is a model with ml_model=None'''
    def __init__(self, *args, **kwargs):
        ml_model = None
        super().__init__(ml_model, *args, **kwargs)
        self.type = 'feature'
        