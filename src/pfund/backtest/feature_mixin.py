from __future__ import annotations

from pfund.backtest.backtest_mixin import BacktestMixin


class BacktestFeatureMixin(BacktestMixin):
    """Backtest behavior specific to features, none yet beyond BacktestMixin.

    Adopted onto a feature via ``BacktestFeatureMixin._adopt(feature)``.
    """
