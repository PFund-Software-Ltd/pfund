# pyright: reportUnknownVariableType=false, reportImplicitAbstractClass=false, reportAbstractUsage=false, reportReturnType=false, reportUnknownParameterType=false
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pfund.typing import FeatureT

from pfund._backtest.backtest_mixin import BacktestMixin


def BacktestFeature(Feature: type[FeatureT], *args: Any, **kwargs: Any) -> FeatureT:
    class _BacktestFeature(BacktestMixin, Feature):
        pass

    _BacktestFeature.__name__ = Feature.__name__
    _BacktestFeature.__qualname__ = Feature.__qualname__
    setattr(_BacktestFeature, "__wrapped__", Feature)

    try:
        return _BacktestFeature(*args, **kwargs)
    except TypeError as e:
        if "__init__()" in str(e):
            raise TypeError(
                f"if super().__init__() is called in {Feature.__name__}.__init__() (which is unnecssary), "
                + "make sure it is called with args and kwargs, i.e. super().__init__(*args, **kwargs)"
            ) from e
        raise
