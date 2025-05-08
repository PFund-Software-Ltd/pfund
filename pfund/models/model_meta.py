from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund.models.model_base import BaseModel, MachineLearningModel
    

from abc import ABCMeta


class MetaModel(ABCMeta):
    def __new__(mcs, name, bases, namespace, **kwargs):
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)
        module_name = namespace.get('__module__', '')
        is_user_defined_class = not module_name.startswith('pfund.')
        if is_user_defined_class:
            original_init = cls.__init__  # capture before overwrite
            def init_in_correct_order(self, *args, **kwargs):
                # force to init the BaseClass first
                BaseClass = cls.__bases__[0]
                BaseClass.__init__(self, *args, **kwargs)
                cls.__original_init__(self, *args, **kwargs)
            cls.__init__ = init_in_correct_order
            cls.__original_init__ = original_init
        return cls

    # NOTE: both __call__ and __init__ will NOT be called when using Ray
    def __init__(cls, name, bases, dct):
        super().__init__(name, bases, dct)
        module_name = dct.get('__module__', '')
        is_user_defined_class = not module_name.startswith('pfund.')
        if is_user_defined_class:
            from pfund.utils.utils import get_args_and_kwargs_from_function
            init_args, _, _, _ = get_args_and_kwargs_from_function(cls.__original_init__)
            # assert users to include 'model'/'indicator' as the first argument in __init__()
            MetaModel._assert_required_arg(cls, init_args)
            
        # FIXME: update backtest model 
        # if name == '_BacktestModel':
        #     assert '__init__' not in dct, '_BacktestModel should not have __init__()'
        
    def __call__(cls, *args, **kwargs):
        model = args[0] if args else kwargs["model"]
        # # TODO: do the same for BaseIndicator, derive if its ta or talib
        ModelClass: type[BaseModel] = cls._derive_model_class(model)
        if not issubclass(cls, ModelClass):
            raise TypeError(
                f"{cls.__name__} using model {model} must inherit from {ModelClass.__name__}, please create your class like this:\n"
                f"class {cls.__name__}(pf.{ModelClass.__name__})"
            )
        instance = super().__call__(*args, **kwargs)
        return instance
        
    @classmethod
    def _assert_required_arg(mcs, cls, init_args: list[str]) -> str:
        BaseClass = cls.__bases__[0]
        if BaseClass.__name__ == 'BaseModel':
            required_arg = 'model'
        elif BaseClass.__name__ == 'BaseIndicator':
            required_arg = 'indicator'
        else:
            required_arg = ''
        if required_arg:
            if required_arg not in init_args or init_args[0] != required_arg:
                raise TypeError(
                    f"{cls.__name__}.__init__() must include the '{required_arg}' as the first argument after 'self', like this:\n"
                    f"{cls.__name__}.__init__(self, {required_arg}, *args, **kwargs)"
                )

    @staticmethod
    def _derive_model_class(model: MachineLearningModel) -> type[BaseModel]:
        try:
            import torch.nn as nn
        except ImportError:
            nn = None

        try:
            from sklearn.base import BaseEstimator
        except ImportError:
            BaseEstimator = None

        if nn is not None and isinstance(model, nn.Module):
            from pfund.models.pytorch_model import PytorchModel
            return PytorchModel
        elif BaseEstimator is not None and isinstance(model, BaseEstimator):
            from pfund.models.sklearn_model import SklearnModel
            return SklearnModel
        else:
            return BaseModel