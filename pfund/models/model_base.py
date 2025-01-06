from __future__ import annotations
from typing import TYPE_CHECKING, Any, Union
if TYPE_CHECKING:
    import pandas as pd
    import polars as pl
    import torch
    import torch.nn as nn
    from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
    from sklearn.pipeline import Pipeline
    from pfund.typing.data import BarDataKwargs, QuoteDataKwargs, TickDataKwargs
    from pfund.models import PytorchModel, SklearnModel
    from pfund.indicators.indicator_base import TaFunction, TalibFunction
    from pfund.typing.literals import tTRADING_VENUE
    from pfund.datas.data_base import BaseData
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

import os
import sys
import inspect
from abc import ABC, abstractmethod
from collections import defaultdict

try:
    import joblib
except ImportError:
    pass
import numpy as np
from rich.console import Console

from pfund.models.model_meta import MetaModel
from pfund.utils.utils import short_path, get_engine_class
from pfund.mixins.trade_mixin import TradeMixin


class BaseModel(TradeMixin, ABC, metaclass=MetaModel):
    def __init__(self, ml_model: MachineLearningModel, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
        self.ml_model = ml_model  # user-defined machine learning model
        self.name = self.mdl = self.get_default_name()
        self.engine = get_engine_class()()
        self._data_tool = self.engine.DataTool()
        self.logger = None
        self._is_running = False
        self._is_ready = defaultdict(bool)  # {data: bool}
        self.datas = defaultdict(dict)  # {product: {repr(resolution): data}}
        self._listeners = defaultdict(list)  # {data: model}
        self._consumers = []  # strategies/models that consume this model
        self._min_data = {}  # {data: int}
        self._max_data = {}  # {data: int}}
        self._num_data = defaultdict(int)  # {data: int}
        self._group_data = True
        self.type = 'model'
        
        self.models = {}
        # NOTE: current model's signal is consumer's prediction
        self.predictions = {}  # {model_name: pred_y}
        self._signals = {}  # {data: signal}, signal = output of predict()
        self._last_signal_ts = {}  # {data: ts}
        self._signal_cols = []
        self._num_signal_cols = 0
        
        self.params = {}
        self.load_params()
        self._assert_predict_function()
    
    @abstractmethod
    def predict(self, X: pd.DataFrame | pl.LazyFrame, *args, **kwargs) -> torch.Tensor | np.ndarray:
        pass

    def _assert_predict_function(self):
        sig = inspect.signature(self.predict)
        params = list(sig.parameters.values())
        if not params or params[0].name != 'X':
            raise Exception(f'{self.name} predict() must have "X" as its first arg, i.e. predict(self, X, *args, **kwargs)')

    def featurize(self) -> pd.DataFrame | pl.LazyFrame:
        Console().print(
            f"WARNING: '{self.name}' is using the default featurize(), "
            "which assumes X = self.df, it could be a wrong input for predict(X).\n"
            f"It is highly recommended to override featurize() in your '{self.name}'.",
            style='bold magenta'
        )
        return self.get_df()
   
    def is_ready(self, data: BaseData) -> bool:
        if not self._is_ready[data]:
            self._num_data[data] += 1
            if self._num_data[data] >= self._min_data[data]:
                self._is_ready[data] = True
        return self._is_ready[data]
    
    def is_model(self) -> bool:
        return not self.is_feature() and not self.is_indicator()
    
    def is_indicator(self) -> bool:
        from pfund.indicators.indicator_base import BaseIndicator
        return isinstance(self, BaseIndicator)
    
    def is_feature(self) -> bool:
        return isinstance(self, BaseFeature)
    
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
    
    def get_model_type_of_ml_model(self) -> PytorchModel | SklearnModel | BaseModel:
        try:
            import torch.nn as nn
        except ImportError:
            nn = None

        try:
            import sklearn
            from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
            from sklearn.pipeline import Pipeline
        except ImportError:
            sklearn = None

        if nn is not None and isinstance(self.ml_model, nn.Module):
            from pfund.models import PytorchModel
            Model = PytorchModel
        elif sklearn is not None and isinstance(self.ml_model, (BaseEstimator, ClassifierMixin, RegressorMixin, Pipeline)):
            from pfund.models import SklearnModel
            Model = SklearnModel
        else:
            Model = BaseModel
        return Model
    
    def set_name(self, name: str):
        self.name = self.mdl = name
    
    def set_min_data(self, min_data: int | dict[BaseData, int]):
        self._min_data = min_data

    def set_max_data(self, max_data: int | dict[BaseData, int]):
        self._max_data = max_data
    
    def set_group_data(self, group_data: bool):
        self._group_data = group_data
    
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
    
    def load(self) -> dict:
        file_path = self._get_file_path()
        if os.path.exists(file_path):
            obj: dict = joblib.load(file_path)
            self.ml_model = obj['ml_model']
            self._assert_no_missing_datas(obj)
            self.logger.debug(f"loaded '{self.name}' from {short_path(file_path)}")
            return obj
        return {}
    
    def dump(self, obj: dict[str, Any] | None=None):
        if obj is None:
            obj = {}
        obj.update({
            'ml_model': self.ml_model,
            'datas': self.datas,
            # TODO: dump dates as well
        })
        file_path = self._get_file_path()
        joblib.dump(obj, file_path, compress=True)
        self.logger.debug(f"dumped '{self.name}' to {short_path(file_path)}")
    
    def add_data(
        self, 
        trading_venue: tTRADING_VENUE, 
        product: str,  # product_basis, defined as {base_asset}_{quote_asset}_{product_type}, e.g. BTC_USDT_PERP
        resolutions: list[str] | str, 
        resamples: dict[str, str] | None=None,
        auto_resample=None,  # FIXME
        quote_data: QuoteDataKwargs | None=None,
        tick_data: TickDataKwargs | None=None,
        bar_data: BarDataKwargs | None=None,
        **product_specs
    ) -> list[BaseData]:
        datas = []
        # consumers must also have model's data
        for consumer in self._consumers:
            for data in consumer.add_data(
                trading_venue, 
                product, 
                resolutions,
                resamples=resamples,
                auto_resample=auto_resample,
                quote_data=quote_data,
                tick_data=tick_data,
                bar_data=bar_data,
                **product_specs
            ):
                self.set_data(data.product, data.resolution, data)
                consumer._add_listener(listener=self, listener_key=data)
                if data not in datas:
                    datas.append(data)
        return datas
    
    def _convert_min_max_data_to_dict(self):
        '''Converts min_data and max_data from int to dict[product, dict[resolution, int]]'''
        DEFAULT_MIN_DATA = 1
        if not self._min_data:
            self._min_data = {data: DEFAULT_MIN_DATA for data in self.get_datas()}
        elif isinstance(self._min_data, int):
            min_data = self._min_data
            self._min_data = {data: min_data for data in self.get_datas()}
            
        if not self._max_data:
            self._max_data = {data: self._min_data[data] for data in self.get_datas()}
        elif isinstance(self._max_data, int):
            max_data = self._max_data
            self._max_data = {data: max_data for data in self.get_datas()}
        
        # check if set up correctly
        for data in self.get_datas():
            assert data in self._min_data, f"{data} not found in {self._min_data=}, make sure set_min_data() is called correctly"
            assert data in self._max_data, f"{data} not found in {self._max_data=}, make sure set_max_data() is called correctly"
    
            min_data = self._min_data[data]
            max_data = self._max_data[data]
            # NOTE: -1 means include all data
            if max_data == -1:
                max_data = sys.maxsize
                
            assert min_data >= 1, f'{min_data=} for {data} must be >= 1'
            assert max_data >= min_data, f'{max_data=} for {data} must be >= {min_data=}'
    
    def _next(self, data: BaseData) -> torch.Tensor | np.ndarray | None:
        '''Returns the next prediction in event-driven manner.'''
        if data in self._last_signal_ts and self._last_signal_ts[data] == data.ts:
            return self._signals[data]
        
        if not self.is_ready(data):
            return None
        
        # if max_data = -1 (include all data), then start_idx = 0
        # if max_data = +x, then start_idx = -x
        start_idx = min(-self._max_data[data], 0)

        if self._group_data:
            product_filter = repr(data.product)
            resolution_filter = data.resol
        else:
            product_filter = resolution_filter = None

        # if group_data, X is per product and resolution -> X[-start_idx:];
        # if not, X is the whole data -> X[-start_idx:]
        X = self.get_df(
            start_idx=start_idx,
            product=product_filter,
            resolution=resolution_filter,
            copy=False
        )
        
        pred_y: torch.Tensor | np.ndarray = self.predict(X)
        new_pred: torch.Tensor | np.ndarray = pred_y[-1]
        if np.isnan(new_pred).all():
            raise Exception(
                f"model '{self.name}' was ready but predicted all NaNs for {data}, \n"
                f"Setting: min_data={self._min_data[data]} (≈ warmup period), max_data={self._max_data[data]}, group_data={self._group_data}, "
                "please make sure it is set up correctly."
            )
        
        self._signals[data] = new_pred
        self._last_signal_ts[data] = data.ts
        
        return new_pred
            
    def start(self):
        if not self.is_running():
            self.add_datas()
            self._add_consumers_datas_if_no_data()
            self._convert_min_max_data_to_dict()
            self.add_models()
            self.add_features()
            self.add_indicators()
            self._start_models()
            self._prepare_df()
            self.load()
            self.on_start()
            self._is_running = True
            self.logger.info(
                f"model '{self.name}' has started.\n"
                f"min_data={self._min_data}\n"
                f"max_data={self._max_data}\n"
                f"group_data={self._group_data}"
            )
        else:
            self.logger.warning(f'model {self.name} has already started')
        
    def stop(self):
        if self.is_running():
            self._is_running = False
            self.on_stop()
            for model in self.models.values():
                model.stop()
        else:
            self.logger.warning(f'model {self.name} has already stopped')
        
    '''
    ************************************************
    Model Functions
    Users can customize these functions in their models.
    ************************************************
    '''
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
        self.set_signal_cols([self.name])
    
    def predict(self, X: pd.DataFrame | pl.LazyFrame, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError
    extract = predict
    