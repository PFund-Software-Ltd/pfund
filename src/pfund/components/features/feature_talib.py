# pyright: reportAttributeAccessIssue=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from narwhals.typing import IntoDataFrame

    from pfund.components.features.feature_base import FeatureValue
    from pfund.typing import ColumnName

import narwhals as nw
import talib
from talib import abstract as talib_abstract
from talib._ta_lib import Function as TALibFunction

from pfund.components.features.feature_base import BaseFeature


class TALibIndicator(BaseFeature):
    def __init__(
        self,
        indicator: TALibFunction | Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ):
        """
        Args:
            indicator:
                A function from either the TA-Lib function or abstract API, e.g.
                ``talib.SMA`` or ``talib.abstract.SMA``.
        """
        if not isinstance(indicator, TALibFunction):
            indicator_name = getattr(indicator, "__name__", None)
            if (
                not indicator_name
                or getattr(talib, indicator_name, None) is not indicator
            ):
                raise ValueError(
                    "indicator must be a TA-Lib function from talib or talib.abstract; e.g. talib.SMA or talib.abstract.SMA"
                )
            try:
                indicator = talib_abstract.Function(indicator_name)
                assert isinstance(indicator, TALibFunction)
            except Exception as exc:
                raise ValueError(
                    f"failed to convert talib.{indicator_name} to an abstract function"
                ) from exc
        self.indicator: TALibFunction = indicator
        super().__init__(indicator, *args, **kwargs)
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
        # NOTE: self.__pfund_args__[0] is the indicator function, so we skip it
        inds = self.indicator(X, *self.__pfund_args__[1:], **self.__pfund_kwargs__)
        inds = nw.from_native(inds, allow_series=True)
        if isinstance(inds, nw.Series):
            return inds.to_native()
        if self._signal_cols and set(self._signal_cols) != set(inds.columns):
            raise ValueError(
                "TA-Lib output columns should NOT be customized: "
                + f"expected {inds.columns}, got {self._signal_cols}"
            )
        return {col: inds[col].to_native() for col in inds.columns}
