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
except ImportError:
    pass
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pfund.strategies.strategy_base import BaseStrategy
    from pfund.models import PytorchModel, SklearnModel
    from pfund.indicators.indicator_base import TAFunction, TALibFunction
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
        TAFunction,  # ta.utils.IndicatorMixin
        TALibFunction,
        Any,
    ]
from pfund.const.paths import MODEL_PATH
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
        self._path = ''
        self._is_load = True  # if True, load trained model from file_path
        self._consumer = None  # strategy/model that consumes this model
        self._is_ready = False
        self._is_running = False
        # minimum number of data required for the model to make a prediction
        self.min_data_points = kwargs.get('min_data_points', 1)
        self.max_data_points = kwargs.get('max_data_points', self.min_data_points)
        assert self.max_data_points >= self.min_data_points, f'max_data_points={self.max_data_points} must be greater than or equal to min_data_points={self.min_data_points}'
        self.ml_model = ml_model  # user-defined machine learning model
        self.signal = None  # output signal df from trained ml_model
        self.products = defaultdict(dict)  # {trading_venue: {pdt1: product1, pdt2: product2} }
        self.datas = defaultdict(dict)  # {product: {'1m': data}}
        self._listeners = defaultdict(list)  # {data: model}
        self.models = {}
        self.predictions = {}
        self.data = None  # last data
        
        self.params = {}
        self.load_params()

    @abstractmethod
    def predict(self, *args, **kwargs) -> pd.DataFrame | torch.Tensor | np.ndarray:
        pass
    
    def __getattr__(self, attr):
        '''gets triggered only when the attribute is not found'''
        if 'ml_model' in self.__dict__ and hasattr(self.ml_model, attr):
            return getattr(self.ml_model, attr)
        else:
            class_name = self.__class__.__name__
            raise AttributeError(f"'{class_name}' object or '{class_name}.ml_model' or '{class_name}.data_tool' has no attribute '{attr}'")
    
    @property
    def df(self):
        return self._data_tool.get_df()
    
    @property
    def data_tool(self):
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
    
    def output_df_to_parquet(self, df, file_path: str):
        self._data_tool.output_df_to_parquet(df, file_path)
    
    # if not specified, features are just the original df
    def prepare_features(self) -> pd.DataFrame:
        return self.df
    
    def set_signal(self, signal: pd.DataFrame | None):
        self.signal = signal
        
    # FIXME: pandas specific
    def to_signal(self, X: pd.DataFrame, pred_y: torch.Tensor | np.ndarray, columns: list[str] | None=None) -> pd.DataFrame:
        if type(pred_y) is torch.Tensor:
            pred_y = pred_y.detach().numpy() if pred_y.requires_grad else pred_y.numpy()
        if not columns:
            num_cols = pred_y.shape[-1]
            if num_cols == 1:
                columns = [self.name]
            else:
                columns = [f'{self.name}_{i}' for i in range(num_cols)]
        signal = pd.DataFrame(pred_y, index=X.index, columns=columns)
        self.set_signal(signal)
        return signal
    
    # FIXME: pandas specific
    def append_to_signal(self, X: pd.DataFrame, new_pred: torch.Tensor | np.ndarray) -> pd.DataFrame:
        '''Appends new signal to self.signal'''
        assert self.signal is not None
        # self.data is the lastest data passed in
        index_data = {'ts': self.data.dt, 'product': repr(self.data.product), 'resolution': repr(self.data.resolution)}
        index = self._data_tool.create_multi_index(index_data, X.index.names)
        new_pred = new_pred.reshape(1, -1)
        signal = pd.DataFrame(new_pred, index=index, columns=self.signal.columns)
        signal = pd.concat([self.signal, signal], ignore_index=False)
        self.set_signal(signal)
        return signal
    
    def flow(self, is_dump=True, path: str='') -> pd.DataFrame:
        X: pd.DataFrame = self.prepare_features()
        pred_y: np.ndarray = self.predict(X)
        # No training
        signal: pd.DataFrame = self.to_signal(X, pred_y)
        if is_dump:
            self.dump(signal, path=path)
        return signal
    
    # FIXME: pandas specific
    def next(self) -> torch.Tensor | np.ndarray | None:
        '''Returns the next prediction in event-driven mode.'''
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
            self.append_to_signal(X, new_pred)
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
    
    def set_consumer(self, consumer: BaseStrategy | BaseModel):
        '''
        when a model is added to a strategy, consumer is a strategy
        when a model is added to a model, consumer is a model
        '''
        self._consumer = consumer

    def create_logger(self):
        self.logger = create_dynamic_logger(self.name, 'model')
        
    def set_name(self, name: str):
        self.name = self.mdl = name
        
    def set_path(self, path: str):
        self._path = path or str(MODEL_PATH)
        
    def set_is_load(self, is_load: bool):
        self._is_load = is_load
    
    def _get_file_path(self, path: str='', extension='.joblib'):
        path = path or self._path
        if os.path.isdir(path):
            path = path[:-1] if path.endswith('/') else path
            file_path = path + f"/{self.name}{extension}"
            self.logger.debug(f"'{short_path(path)}' is a folder, derive file_path={short_path(file_path)}")
        elif os.path.isfile(path):
            file_path = path
        else:
            file_path = f"{MODEL_PATH}/{self.name}{extension}"
        return file_path
    
    def _assert_no_missing_datas(self, obj):
        loaded_datas = {data for product in obj['datas'] for data in obj['datas'][product].values()}
        added_datas = {data for product in self.datas for data in self.datas[product].values()}
        if loaded_datas != added_datas:
            missing_datas = loaded_datas - added_datas
            raise Exception(f"missing data {missing_datas} in model '{self.name}', please use add_data() to add them back")
    
    def load(self, path: str=''):
        file_path = self._get_file_path(path=path)
        if os.path.exists(file_path):
            obj = joblib.load(file_path)
            signal = obj['signal']
            self.set_signal(signal)
            self.ml_model = obj['ml_model']
            self._assert_no_missing_datas(obj)
            self.logger.debug(f"loaded trained ml_model '{self.name}' and its signal from {short_path(file_path)}")
        else:
            self.logger.debug(f"no trained ml_model '{self.name}' found in {short_path(file_path)}")
    
    def dump(self, signal: pd.DataFrame, path: str=''):
        obj = {
            'signal': signal,
            'ml_model': self.ml_model,
            'datas': self.datas,
            # TODO: dump dates as well
        }
        file_path = self._get_file_path(path=path)
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
        return self._is_ready
    
    def is_running(self):
        return self._is_running
    
    def is_indicator(self) -> bool:
        from pfund.indicators.indicator_base import BaseIndicator
        return isinstance(self, BaseIndicator)
    
    def _is_prepared_signal_required(self):
        return True
    
    def get_datas(self) -> list[BaseData]:
        datas = []
        for product in self.datas:
            datas.extend(list(self.datas[product].values()))
        return datas
    
    def get_data(self, product: BaseProduct, resolution: str | None=None):
        return self.datas[product] if not resolution else self.datas[product][resolution]
    
    def add_data(self, trading_venue, base_currency, quote_currency, ptype, *args, **kwargs) -> list[BaseData]:
        datas = self._consumer.add_data(trading_venue, base_currency, quote_currency, ptype, *args, **kwargs)
        for data in datas:
            self._add_data(data)
        return datas
    
    def _add_data(self, data: BaseData):
        self.datas[data.product][repr(data.resolution)] = data
        self._consumer.add_listener(listener=self, listener_key=data)
    
    def _add_consumer_datas_if_no_data(self) -> list[BaseData]:
        '''if no data, add the consumer's datas'''
        if self.datas:
            return []
        else:
            self.logger.warning(f'No data for model {self.name}, adding {self._consumer.name}\'s datas')
            consumer_datas = self._consumer.get_datas()
            for data in consumer_datas:
                self._add_data(data)
            return consumer_datas
    
    # IMPROVE: current issue is in next(), when the df has multiple products and resolutions,
    # don't know how to determine the exact minimum amount of data points for predict()
    def _adjust_min_data_points(self):
        num_products = len(self.get_df_products())
        num_resolutions = len(self.get_df_resolutions())
        assert num_products and num_resolutions, f"{num_products=} and/or {num_resolutions=} are invalid, please check your dataframe"
        adj_min_data_points = self.min_data_points * num_products * num_resolutions
        adj_max_data_points = self.max_data_points * num_products * num_resolutions
        self.logger.warning(f'adjust min_data_points from {self.min_data_points} to {adj_min_data_points} with {num_products=} and {num_resolutions=}')
        self.logger.warning(f'adjust max_data_points from {self.max_data_points} to {adj_max_data_points} with {num_products=} and {num_resolutions=}')
        self.min_data_points = adj_min_data_points
        self.max_data_points = adj_max_data_points
        
    def get_model(self, name: str) -> BaseModel:
        return self.models[name]
    
    def add_model(self, model: tModel, name: str='', model_path: str='', is_load=True) -> tModel:
        Model = model.get_model_type_of_ml_model()
        assert isinstance(model, Model), \
            f"model '{model.__class__.__name__}' is not an instance of {Model.__name__}. Please create your model using 'class {model.__class__.__name__}({Model.__name__})'"
        if name:
            model.set_name(name)
        model.set_path(model_path)
        model.create_logger()
        mdl = model.mdl
        if mdl in self.models:
            raise Exception(f"model '{mdl}' already exists in model '{self.name}'")
        model.set_consumer(self)
        model.set_is_load(is_load)
        self.models[mdl] = model
        self.logger.debug(f"added model '{mdl}'")
        return model
    
    def add_feature(self, feature: tFeature, name: str='', feature_path: str='', is_load: bool=True) -> tFeature:
        return self.add_model(feature, name=name, model_path=feature_path, is_load=is_load)
    
    def add_indicator(self, indicator: tIndicator, name: str='', indicator_path: str='', is_load: bool=True) -> tIndicator:
        return self.add_model(indicator, name=name, model_path=indicator_path, is_load=is_load)
    
    def update_quote(self, data: QuoteData, **kwargs):
        product, bids, asks, ts = data.product, data.bids, data.asks, data.ts
        self.data = data
        for listener in self._listeners[data]:
            model = listener
            model.update_quote(data, **kwargs)
            self.update_predictions(model)
        self._append_to_df(data, **kwargs)
        self.on_quote(product, bids, asks, ts, **kwargs)
        
    def update_tick(self, data: TickData, **kwargs):
        product, px, qty, ts = data.product, data.px, data.qty, data.ts
        self.data = data
        for listener in self._listeners[data]:
            model = listener
            model.update_tick(data, **kwargs)
            self.update_predictions(model)
        self._append_to_df(data, **kwargs)
        self.on_tick(product, px, qty, ts, **kwargs)
    
    def update_bar(self, data: BarData, **kwargs):
        product, bar, ts = data.product, data.bar, data.bar.end_ts
        self.data = data
        for listener in self._listeners[data]:
            model = listener
            model.update_bar(data, **kwargs)
            self.update_predictions(model)
        self._append_to_df(data, self.predictions, **kwargs)
        self.on_bar(product, bar, ts, **kwargs)
    
    def update_predictions(self, model: BaseModel):
        pred_y: torch.Tensor | np.ndarray | None = model.next()
        self.predictions[model.name] = pred_y
    
    def _start_models(self):
        for model in self.models.values():
            model.start()

    def _prepare_df(self):
        return self._data_tool.prepare_df()
        
    def _prepare_df_with_models(self):
        return self._data_tool.prepare_df_with_models(self.models)
    
    def _append_to_df(self, data, predictions, **kwargs):
        return self._data_tool.append_to_df(data, predictions, **kwargs)
    
    def start(self):
        if not self.is_running():
            self.add_datas()
            self._add_consumer_datas_if_no_data()
            self.add_models()
            self._start_models()
            self._prepare_df()
            if self._is_load:
                self.load()  # load trained model
            # prepare indicator's signal on the fly if required
            if self._is_prepared_signal_required() and self.signal is None:
                if self.is_indicator():
                    self.logger.debug(f'calculating indicator {self.name} signal(s) on the fly')
                    self.flow(is_dump=False)
                else:
                    raise Exception(f"signal is None, please make sure model '{self.name}' is loaded or was dumped using 'model.dump(signal)' correctly.")
            self._prepare_df_with_models()
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
        