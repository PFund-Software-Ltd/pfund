from __future__ import annotations
from typing import TYPE_CHECKING, Any, Union
if TYPE_CHECKING:
    from torch import Tensor
    from numpy import ndarray
    import torch.nn as nn
    from sklearn.base import BaseEstimator
    from pfeed.typing import GenericFrame
    from pfund.indicators.indicator_base import TaFunction, TalibFunction
    MachineLearningModel = Union[
        nn.Module,
        BaseEstimator,
        TaFunction,  # ta.utils.IndicatorMixin
        TalibFunction,
    ]
    from pfund.datas.data_base import BaseData

import os
from abc import ABC, abstractmethod

from pfund.models.model_meta import MetaModel
from pfund.mixins.trade_mixin import TradeMixin


class BaseModel(TradeMixin, ABC, metaclass=MetaModel):
    def __init__(self, model: MachineLearningModel, *args, **kwargs):
        from collections import defaultdict
        self.model = model  # user-defined machine learning model

        self._min_data = {}  # {data: int}
        self._max_data = {}  # {data: int}
        self._num_data = defaultdict(int)  # {data: int}
        self._group_data = True
        
        self.__mixin_post_init__(model, *args, **kwargs)  # calls TradeMixin.__mixin_post_init__()
    
    @abstractmethod
    def predict(self, X: GenericFrame, *args, **kwargs) -> Tensor | ndarray:
        pass
    
    def _assert_functions_signatures(self):
        from pfund.utils.utils import get_args_and_kwargs_from_function
        super()._assert_functions_signatures()
        def _assert_predict_function():
            args, kwargs, _, _ = get_args_and_kwargs_from_function(self.predict)
            if not args or args[0] != 'X':
                raise Exception(f'{self.name} predict() must have "X" as its first arg, i.e. predict(self, X, *args, **kwargs)')
        _assert_predict_function()

    def featurize(self) -> GenericFrame:
        from pfund import cprint
        cprint(
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
    
    def _set_min_data(self, min_data: int | dict[BaseData, int]):
        self._min_data = min_data

    def _set_max_data(self, max_data: int | dict[BaseData, int]):
        self._max_data = max_data
    
    def _set_group_data(self, group_data: bool):
        self._group_data = group_data
    
    def _get_file_path(self, extension='.joblib'):
        from pfund import get_config
        config = get_config()
        path = f'{config.artifact_path}/{self.name}'
        file_name = f'{self.name}{extension}'
        if not os.path.exists(path):
            os.makedirs(path)
        return f"{path}/{file_name}"
    
    def _assert_no_missing_datas(self, obj):
        loaded_datas = {data for product in obj['datas'] for data in obj['datas'][product].values()}
        added_datas = {data for product in self._datas for data in self._datas[product].values()}
        if loaded_datas != added_datas:
            missing_datas = loaded_datas - added_datas
            raise Exception(f"missing data {missing_datas} in model '{self.name}', please use add_data() to add them back")
    
    def load(self) -> dict:
        import joblib
        from pfund.utils.utils import short_path
        file_path = self._get_file_path()
        if os.path.exists(file_path):
            obj: dict = joblib.load(file_path)
            self.model = obj['model']
            self._assert_no_missing_datas(obj)
            self.logger.debug(f"loaded '{self.name}' from {short_path(file_path)}")
            return obj
        return {}
    
    def dump(self, obj: dict[str, Any] | None=None):
        import joblib
        from pfund.utils.utils import short_path
        if obj is None:
            obj = {}
        obj.update({
            'model': self.model,
            'datas': self._datas,
            # TODO: dump dates as well
        })
        file_path = self._get_file_path()
        joblib.dump(obj, file_path, compress=True)
        self.logger.debug(f"dumped '{self.name}' to {short_path(file_path)}")
    
    def _convert_min_max_data_to_dict(self):
        '''Converts min_data and max_data from int to dict[product, dict[resolution, int]]'''
        from sys import maxsize
        DEFAULT_MIN_DATA = 1
        if not self._min_data:
            self._min_data = {data: DEFAULT_MIN_DATA for data in self.list_datas()}
        elif isinstance(self._min_data, int):
            min_data = self._min_data
            self._min_data = {data: min_data for data in self.list_datas()}
            
        if not self._max_data:
            self._max_data = {data: self._min_data[data] for data in self.list_datas()}
        elif isinstance(self._max_data, int):
            max_data = self._max_data
            self._max_data = {data: max_data for data in self.list_datas()}
        
        # check if set up correctly
        for data in self.list_datas():
            assert data in self._min_data, f"{data} not found in {self._min_data=}, make sure _set_min_data() is called correctly"
            assert data in self._max_data, f"{data} not found in {self._max_data=}, make sure _set_max_data() is called correctly"
    
            min_data = self._min_data[data]
            max_data = self._max_data[data]
            # NOTE: -1 means include all data
            if max_data == -1:
                max_data = maxsize
                
            assert min_data >= 1, f'{min_data=} for {data} must be >= 1'
            assert max_data >= min_data, f'{max_data=} for {data} must be >= {min_data=}'
    
    def _next(self, data: BaseData) -> Tensor | ndarray | None:
        '''Returns the next prediction in event-driven manner.'''
        from numpy import isnan
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
        
        pred_y: Tensor | ndarray = self.predict(X)
        new_pred: Tensor | ndarray = pred_y[-1]
        if isnan(new_pred).all():
            raise Exception(
                f"model '{self.name}' was ready but predicted all NaNs for {data}, \n"
                f"Setting: min_data={self._min_data[data]} (≈ warmup period), max_data={self._max_data[data]}, group_data={self._group_data}, "
                "please make sure it is set up correctly."
            )
        
        self._signals[data] = new_pred
        self._last_signal_ts[data] = data.ts
        
        return new_pred
            
    def start(self):
        super().start()
        self._convert_min_max_data_to_dict()
        self.load()
        self.logger.info(
            f"model '{self.name}' has started.\n"
            f"min_data={self._min_data}\n"
            f"max_data={self._max_data}\n"
            f"group_data={self._group_data}"
        )
        
    def stop(self, reason: str=''):
        super().stop(reason=reason)
