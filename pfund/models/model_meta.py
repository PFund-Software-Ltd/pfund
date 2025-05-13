from __future__ import annotations
from typing import TYPE_CHECKING, Literal
if TYPE_CHECKING:
    from pfund.models.model_base import BaseModel, MachineLearningModel
    from pfund.indicators.indicator_base import BaseIndicator
    from pfund.indicators.indicator_base import IndicatorFunction
    

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
    
    @staticmethod
    def _get_required_arg(cls) -> Literal['model', 'indicator', '']:
        from pfund.models.model_base import BaseModel
        from pfund.indicators.indicator_base import BaseIndicator
        from pfund.features.feature_base import BaseFeature
        if issubclass(cls, BaseFeature):
            return ''
        elif issubclass(cls, BaseIndicator):
            return 'indicator'
        elif issubclass(cls, BaseModel):
            return 'model'
        else:
            return ''
        
    @classmethod
    def _assert_required_arg(mcs, cls, init_args: list[str]) -> str:
        BaseClass = cls.__bases__[0]
        required_arg = cls._get_required_arg(BaseClass)
        if required_arg:
            if required_arg not in init_args or init_args[0] != required_arg:
                raise TypeError(
                    f"{cls.__name__}.__init__() must include `{required_arg}` as the first argument after `self`, like this:\n"
                    f"{cls.__name__}.__init__(self, {required_arg}, *args, **kwargs)"
                )
                
    def __call__(cls, *args, **kwargs):
        if required_arg := cls._get_required_arg(cls):
            # required_component could be model or indicator
            required_component = args[0] if args else kwargs.get(required_arg, None)
            if required_component is not None:
                if required_arg == 'model':
                    ParentClass: type[BaseModel] = cls._derive_model_class(required_component)
                elif required_arg == 'indicator':
                    ParentClass: type[BaseIndicator] = cls._derive_indicator_class(required_component)
                else:
                    raise ValueError(f"Unsupported required_arg: {required_arg}")
                if not issubclass(cls, ParentClass):
                    raise TypeError(
                        f"'{cls.__name__}' must inherit from '{ParentClass.__name__}' when arg `{required_arg}` is of type '{type(required_component)}', please create your class like this:\n"
                        f"class {cls.__name__}(pf.{ParentClass.__name__})"
                    )
        instance = super().__call__(*args, **kwargs)
        return instance

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
            from pfund.models.model_base import BaseModel
            return BaseModel
    
    @staticmethod
    def _derive_indicator_class(indicator: IndicatorFunction) -> type[BaseIndicator]:
        from pfund.indicators.talib_indicator import TalibFunction
        if isinstance(indicator, TalibFunction):
            from pfund.indicators.talib_indicator import TalibIndicator
            return TalibIndicator
        else:
            from pfund.indicators.ta_indicator import TaIndicator
            return TaIndicator