# pyright: reportAttributeAccessIssue=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from narwhals.typing import IntoDataFrame

    from pfund.components.features.feature_base import FeatureValue
    from pfund.typing import ColumnName

import narwhals as nw
from talib._ta_lib import Function as TalibFunction

from pfund.components.features.feature_base import BaseFeature


class TalibIndicator(BaseFeature):
    def __init__(self, indicator: TalibFunction, *args: Any, **kwargs: Any):
        """
        Args:
            indicator:
                from talib import abstract as talib
                e.g. indicator = talib.SMA
        """
        if not isinstance(indicator, TalibFunction):
            raise ValueError(
                "indicator must be a talib function, e.g. from talib.abstract import SMA"
            )
        super().__init__(*args, **kwargs)
        self.indicator: TalibFunction = indicator
        if default_params := {
            k: v
            for k, v in self.get_indicator_params().items()
            if k not in self.__pfund_kwargs__
        }:
            self.logger.warning(
                f"talib indicator {self.name} is using default parameters {default_params}"
            )
        timeperiod = self.__pfund_kwargs__.get(
            "timeperiod", None
        ) or default_params.get("timeperiod", None)
        if timeperiod:
            self.config["lookback_period"] = int(timeperiod)

    def _get_default_name(self) -> str:
        return self.get_indicator_name() or super()._get_default_name()

    def get_indicator_info(self):
        return self.indicator.info

    def get_indicator_name(self):
        return self.get_indicator_info()["name"]

    def get_indicator_params(self):
        return self.get_indicator_info()["parameters"]

    def transform(
        self, X: IntoDataFrame
    ) -> FeatureValue | dict[ColumnName, FeatureValue]:
        # talib.abstract accepts pandas/polars frames directly and returns the same type
        inds = self.indicator(X, *self.__pfund_args__, **self.__pfund_kwargs__)
        if len(self.indicator.output_names) == 1:
            # single-output indicator (e.g. SMA): returns a Series -> one column named self.name
            return inds
        # multi-output indicator (e.g. BBANDS): returns a frame -> dict keys become "{self.name}-{output_name}"
        inds = nw.from_native(inds)
        return {col: inds[col].to_native() for col in inds.columns}
