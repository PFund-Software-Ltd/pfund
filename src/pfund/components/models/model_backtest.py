# pyright: reportUnknownVariableType=false, reportImplicitAbstractClass=false, reportAbstractUsage=false, reportReturnType=false, reportUnknownParameterType=false
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pfund.components.models.model_base import UnderlyingModel
    from pfund.typing import ModelT

from pfund._backtest.backtest_mixin import BacktestMixin
from pfund.components.features.feature_base import BaseFeature


def BacktestModel(
    Model: type[ModelT], model: UnderlyingModel, *args: Any, **kwargs: Any
) -> ModelT:
    class _BacktestModel(BacktestMixin, Model):
        _is_training: bool = False

        def is_training(self) -> bool:
            return self._is_training

        def _materialize(self):
            super()._materialize()
            try:
                self.load()
            except FileNotFoundError:
                from pfund_kit.style import cprint, TextStyle, RichColor

                cprint(
                    f"No trained model found for '{self.name}' — treating this run as a training session. "
                    + "Fit your model, then call save() so future runs can load it and generate predictions.",
                    style=TextStyle.BOLD + RichColor.YELLOW,
                )
                self._is_training = True

    try:
        if not issubclass(Model, BaseFeature):
            return _BacktestModel(model, *args, **kwargs)
        else:
            return _BacktestModel(*args, **kwargs)
    except TypeError as e:
        if "__init__()" in str(e):
            raise TypeError(
                f"if super().__init__() is called in {Model.__name__}.__init__() (which is unnecssary), "
                + "make sure it is called with args and kwargs, i.e. super().__init__(*args, **kwargs)"
            ) from e
        raise
