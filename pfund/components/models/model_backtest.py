# pyright: reportUnknownVariableType=false, reportImplicitAbstractClass=false, reportAbstractUsage=false, reportReturnType=false, reportUnknownParameterType=false
from __future__ import annotations
from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    from pfund.components.models.model_base import MachineLearningModel
    from pfund.typing import ModelT

from pfund.components.features.feature_base import BaseFeature
from pfund._backtest.backtest_mixin import BacktestMixin


def BacktestModel(Model: type[ModelT], model: MachineLearningModel, *args: Any, **kwargs: Any) -> ModelT:
    class _BacktestModel(BacktestMixin, Model):
        # TODO: catch exception, allow failure loading a model if it hasn't been dumped yet
        def load(self):
            return super().load()
        
    try:       
        if not issubclass(Model, BaseFeature):
            return _BacktestModel(model, *args, **kwargs)
        else:
            return _BacktestModel(*args, **kwargs)
    except TypeError as e:
        if '__init__()' in str(e):
            raise TypeError(
                f'if super().__init__() is called in {Model.__name__}.__init__() (which is unnecssary), ' +
                'make sure it is called with args and kwargs, i.e. super().__init__(*args, **kwargs)'
            ) from e
        raise