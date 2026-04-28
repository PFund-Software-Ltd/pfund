# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false, reportUnknownVariableType=false, reportAssignmentType=false, reportArgumentType=false, reportUnknownParameterType=false
from __future__ import annotations
from typing import TYPE_CHECKING, Any, cast, Literal
if TYPE_CHECKING:
    from torch import Tensor
    from sklearn.base import BaseEstimator
    from numpy import ndarray
    import torch.nn as nn
    from narwhals._native import NativeDataFrame
    from pfund.typing import ModelT, ColumnName
    from pfund.components.actor_proxy import ActorProxy
    from pfund.enums import TradingVenue
    from pfund.datas.data_market import MarketData
    from pfund.datas.data_config import DataConfig
    from pfeed.storages.storage_config import StorageConfig
    from pfund.components.indicators.indicator_base import TalibFunction
    MachineLearningModel = nn.Module | BaseEstimator | TalibFunction

from abc import ABC, abstractmethod

import narwhals as nw

from pfund.components.models.model_meta import MetaModel
from pfund.components.mixin import ComponentMixin
from pfund.enums import ArtifactType


def wrap_model(model: BaseEstimator | nn.Module | ModelT | ActorProxy[ModelT]) -> ModelT | ActorProxy[ModelT]:
    '''Wraps a sklearn/pytorch model into a pfund model'''
    from pfund.components.actor_proxy import ActorProxy
    from pfund.components.models.model_base import BaseModel
    if isinstance(model, (ActorProxy, BaseModel)):
        return model
    try:
        from sklearn.base import BaseEstimator
        if isinstance(model, BaseEstimator):
            from pfund.components.models.sklearn_model import SklearnModel
            pfund_model = SklearnModel(model)
            pfund_model._set_name(type(model).__name__)
            return cast("ModelT", pfund_model)
    except ImportError:
        pass
    try:
        import torch.nn as nn
        if isinstance(model, nn.Module):
            from pfund.components.models.pytorch_model import PytorchModel
            pfund_model = PytorchModel(model)
            pfund_model._set_name(type(model).__name__)
            return cast("ModelT", pfund_model)
    except ImportError:
        pass
    raise TypeError(f"Unsupported model type: {type(model).__name__}")


class BaseModel(ComponentMixin, ABC, metaclass=MetaModel):
    def __init__(self, model: MachineLearningModel, *args: Any, **kwargs: Any):
        self.model = model  # user-defined machine learning model
        self._df_form: Literal['wide', 'long'] = 'wide'
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
            raise AttributeError(f"'{self.name}' has no attribute '{name}'")
        try:
            return getattr(model, name)
        except AttributeError:
            raise AttributeError(f"'{self.name}' and its underlying model '{self.model.__class__.__name__}' both have no attribute '{name}'")

    def signalize(self, features_df: NativeDataFrame) -> dict[ColumnName, Any]:
        '''Creates signals_df (combined signals from other component)
        Args:
            data_df: dataframe in {self._df_form} form
        '''
        X = nw.from_native(features_df)
        pred: Tensor | ndarray = self.predict(X)
        is_from_pytorch = type(pred).__module__.startswith('torch')
        if is_from_pytorch:
            pred = pred.detach().cpu().numpy()
        signal_cols = self._signal_cols
        num_signal_cols = len(signal_cols)
        if pred.ndim == 1:
            if num_signal_cols != 1:
                raise ValueError(f"prediction is 1D but {self.name} has {num_signal_cols} signal columns: {signal_cols}")
            signals_dict = {signal_cols[0]: pred}
        else:
            # last dimension = signal columns, everything in between is packed into cells
            if num_signal_cols != pred.shape[-1]:
                raise ValueError(f"Expected {num_signal_cols} signal columns for {self.name}, but prediction has shape {pred.shape}")
            signals_dict = {}
            for i, col in enumerate(signal_cols):
                values = pred[..., i]
                # NOTE: list() converts 2D+ array into per-row sub-arrays, needed because pandas rejects >1D per-column arrays
                signals_dict[col] = list(values) if values.ndim > 1 else values
        return signals_dict

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            'model': self.model.__class__.__name__,
        }
    
    def add_data(
        self, 
        venue: TradingVenue | str,
        product: str,
        exchange: str='',
        symbol: str='',
        product_name: str='',
        data_config: DataConfig | None=None,
        storage_config: StorageConfig | None=None,
        **product_specs: Any
    ) -> list[MarketData]:
        datas: list[MarketData] = super().add_data(
            venue=venue,
            product=product,
            exchange=exchange,
            symbol=symbol,
            product_name=product_name,
            data_config=data_config,
            storage_config=storage_config,
            **product_specs
        )
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
    
    def _assert_no_missing_datas(self, obj):
        loaded_datas = {data for product in obj['datas'] for data in obj['datas'][product].values()}
        added_datas = {data for product in self._datas for data in self._datas[product].values()}
        if loaded_datas != added_datas:
            missing_datas = loaded_datas - added_datas
            raise Exception(f"missing data {missing_datas} in model '{self.name}', please use add_data() to add them back")
    
    def load(self):
        # TEMP
        # from pfund_kit.logging.filters.trimmed_path_filter import TrimmedPathFilter
        # trim_path = TrimmedPathFilter.trim_path
        # TODO:
        # self.store.component_feed.retrieve(
        #     artifact_type=ArtifactType.model,
        #     storage=self.storage_config.storage,
        #     data_path=self.context.pfund_config.data_path,
        # )
        # TEMP
        # obj: dict = joblib.load(file_path)
        # self.model = obj['model']
        # self._assert_no_missing_datas(obj)
        return
    
    def dump(self):
        from pfeed.enums.data_storage import FileBasedDataStorage
        if self.storage_config is not None:
            storage = self.storage_config.storage
        else:
            default_storage = FileBasedDataStorage.LOCAL
            storage = default_storage.value
        self.store.component_feed.load(
            artifact_type=ArtifactType.model,
            storage=storage,
            data_path=self.context.pfund_config.data_path,
        )
