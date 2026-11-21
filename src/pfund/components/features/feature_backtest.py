from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pfund.typing import FeatureT

from pfund.backtest.backtest_mixin import BacktestMixin


def BacktestFeature(feature: FeatureT) -> FeatureT:
    """Turns `feature` into its backtesting version in place, keeping its state."""
    # already adopted, e.g. the same feature instance shared by two parents
    if isinstance(feature, BacktestMixin):
        return feature
    Feature = type(feature)

    class _BacktestFeature(BacktestMixin, Feature):
        pass

    _BacktestFeature.__name__ = Feature.__name__
    _BacktestFeature.__qualname__ = Feature.__qualname__
    setattr(_BacktestFeature, "__wrapped__", Feature)

    return _BacktestFeature._adopt(feature)
