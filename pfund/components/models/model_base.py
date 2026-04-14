# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false, reportUnknownVariableType=false, reportAssignmentType=false, reportArgumentType=false
from __future__ import annotations
from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    from torch import Tensor
    from sklearn.base import BaseEstimator
    from numpy import ndarray
    import torch.nn as nn
    from narwhals._native import NativeDataFrame
    from pfund.enums import TradingVenue
    from pfund.datas.data_market import MarketData
    from pfund.datas.data_config import DataConfig
    from pfeed.storages.storage_config import StorageConfig
    from pfund.components.indicators.indicator_base import TalibFunction
    MachineLearningModel = nn.Module | BaseEstimator | TalibFunction
    from pfund.datas.data_base import BaseData

import os
from abc import ABC, abstractmethod

import narwhals as nw

from pfund_kit.logging.filters.trimmed_path_filter import TrimmedPathFilter
from pfund.components.models.model_meta import MetaModel
from pfund.components.mixin import ComponentMixin


trim_path = TrimmedPathFilter.trim_path


class BaseModel(ComponentMixin, ABC, metaclass=MetaModel):
    def __init__(self, model: MachineLearningModel, *args: Any, **kwargs: Any):
        from collections import defaultdict
        self.model = model  # user-defined machine learning model
        self._num_data = defaultdict(int)  # {data: int}
        
        self.__mixin_post_init__(model, *args, **kwargs)  # calls ComponentMixin.__mixin_post_init__()
    
    @abstractmethod
    def predict(self, X: Any, *args: Any, **kwargs: Any) -> Any:
        pass

    def __getattr__(self, name: str):
        '''
        __getattr__ only fires when the attribute isn't found via normal lookup,
        so we only reach here for methods/attrs not defined on the wrapper.
        '''
        # Skip underscore names: 
        # (1) dunders like __reduce__/__getstate__ have
        # pickle-protocol meaning and must not silently resolve to self.model,
        # (2) single-underscore names are pfund-internal state — delegating them
        # risks silent collisions if the underlying model grows a same-named attr,
        # (3) avoids recursion during __init__/unpickle when private attrs may be
        # accessed before they're set on self.
        if name.startswith('_'):
            raise AttributeError(name)
        model = self.__dict__.get('model')
        if model is None:
            raise AttributeError(name)
        return getattr(model, name)

    def signalize(self, features_df: NativeDataFrame) -> NativeDataFrame:
        '''Creates signals_df (combined signals from other component)
        Args:
            data_df: dataframe in {self.df_form} form
        '''
        X = nw.from_native(features_df)
        pred_y: Tensor | ndarray = self.predict(X)
        is_from_pytorch = type(pred_y).__module__.startswith('torch')
        if is_from_pytorch:
            pred_y = pred_y.detach().cpu().numpy()  
        signal_cols = self.get_signal_cols()
        num_signal_cols = len(signal_cols)
        df_backend = nw.get_native_namespace(features_df)
        if pred_y.ndim == 1:
            if num_signal_cols != 1:
                raise ValueError(f"pred_y is 1D but {self.name} has {num_signal_cols} signal columns: {signal_cols}")
            signals_dict = {signal_cols[0]: pred_y}
        else:
            # last dimension = signal columns, everything in between is packed into cells
            if num_signal_cols != pred_y.shape[-1]:
                raise ValueError(f"Expected {num_signal_cols} signal columns for {self.name}, but pred_y has shape {pred_y.shape}")
            signals_dict = {}                                                                                                         
            for i, col in enumerate(signal_cols):                                                                                     
                values = pred_y[..., i]                                                                                               
                # NOTE: list() converts 2D+ array into per-row sub-arrays, needed because pandas rejects >1D per-column arrays
                signals_dict[col] = list(values) if values.ndim > 1 else values
        signals_df = nw.DataFrame.from_dict(signals_dict, backend=df_backend)
        return signals_df.to_native()

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            'model': self.model.__class__.__name__,
        }
    
    def add_data(
        self, 
        trading_venue: TradingVenue | str,
        product: str,
        exchange: str='',
        symbol: str='',
        product_name: str='',
        data_config: DataConfig | None=None,
        storage_config: StorageConfig | None=None,
        warmup_period: int | None=None,
        lookback_period: int | None=None,
        **product_specs: Any
    ) -> list[MarketData]:
        datas: list[MarketData] = super().add_data(
            trading_venue=trading_venue,
            product=product,
            exchange=exchange,
            symbol=symbol,
            product_name=product_name,
            data_config=data_config,
            storage_config=storage_config,
            **product_specs
        )
        self.databoy.market_data_store.set_periods(warmup_period, lookback_period)
        return datas

    def _assert_functions_signatures(self):
        from pfund_kit.utils.function import get_function_args_and_kwargs
        super()._assert_functions_signatures()
        def _assert_predict_function():
            args, kwargs, _, _ = get_function_args_and_kwargs(self.predict)
            if not args or args[0] != 'X':
                raise Exception(f'{self.name} predict() must have "X" as its first arg, i.e. predict(self, X, *args, **kwargs)')
        _assert_predict_function()
    
    def _get_default_signal_cols(self, num_cols: int) -> list[str]:
        if num_cols == 1:
            columns = [self.name]
        else:
            columns = [f'{self.name}-{i}' for i in range(num_cols)]
        return columns
    
    def is_ready(self, data: BaseData) -> bool:
        if not self._is_ready[data]:
            self._num_data[data] += 1
            if self._num_data[data] >= self._min_data[data]:
                self._is_ready[data] = True
        return self._is_ready[data]
    
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
        file_path = self._get_file_path()
        if os.path.exists(file_path):
            obj: dict = joblib.load(file_path)
            self.model = obj['model']
            self._assert_no_missing_datas(obj)
            self.logger.debug(f"loaded '{self.name}' from {trim_path(file_path)}")
            return obj
        return {}
    
    def dump(self, obj: dict[str, Any] | None=None):
        import joblib
        if obj is None:
            obj = {}
        obj.update({
            'model': self.model,
            'datas': self._datas,
            # TODO: dump dates as well
        })
        file_path = self._get_file_path()
        joblib.dump(obj, file_path, compress=True)
        self.logger.debug(f"dumped '{self.name}' to {trim_path(file_path)}")
    
    # FIXME: DEPRECATED?
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
        # FIXME
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

        # FIXME: should add featurize() to get X, feature_df != data_df
        
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
