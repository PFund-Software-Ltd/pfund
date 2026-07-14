from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeAlias, cast
from collections.abc import Sequence

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from torch import Tensor
    from narwhals.typing import IntoDataFrame, IntoSeries

    from pfund.typing import ColumnName, Signals

    FeatureValue: TypeAlias = "NDArray[Any] | Tensor | Sequence[Any] | IntoSeries"

from abc import ABC, abstractmethod

import numpy as np

from pfund.enums import ComponentType
from pfund.components.mixin import ComponentMixin
from pfund.components.features.feature_meta import MetaFeature


class BaseFeature(ComponentMixin, ABC, metaclass=MetaFeature):
    def __init__(self, *args: Any, **kwargs: Any):
        self.component_type = ComponentType.feature
        self._df_form: Literal["wide", "long"] = "wide"
        self.__mixin_post_init__(
            *args, **kwargs
        )  # calls ComponentMixin.__mixin_post_init__()

    @abstractmethod
    def transform(
        self,
        X: IntoDataFrame,
        *args: Any,
        **kwargs: Any,
    ) -> FeatureValue | dict[ColumnName, FeatureValue]:
        """Extract features from the input data"""
        pass

    def signalize(self, X: IntoDataFrame) -> Signals:
        """Creates signals of this component

        Args:
            X: features df

        Returns:
            dict[ColumnName, Any]: The predicted signals.
        """
        features = self.transform(X)
        if features is None:
            raise ValueError(
                f"{self.name} transform() returned None, did you forget the return statement?"
            )

        if not self._signal_cols:
            num_cols = len(features) if isinstance(features, dict) else 1
            signal_cols = self._get_default_signal_cols(num_cols=num_cols)
            self.set_signal_cols(signal_cols)

        if not isinstance(features, dict):
            if len(self._signal_cols) > 1:
                raise ValueError(
                    f"Expected a dict returend from transform(), got {features}"
                )
            features = {self._signal_cols[0]: features}

        if set(features) != set(self._signal_cols):
            raise ValueError(
                f"{self.name} transform() output columns changed: "
                + f"expected {self._signal_cols}, got {list(features)}"
            )

        signals: Signals = {}
        for col, value in features.items():
            is_from_pytorch = type(value).__module__.startswith("torch")
            if is_from_pytorch:
                value = cast("Tensor", value).detach().cpu().numpy()
            value = np.asarray(value)  # coerces Series/list; no copy if already ndarray
            if value.ndim == 0:
                raise ValueError(
                    f"{self.name} transform() returned a scalar for '{col}'; "
                    + "expected one value per row"
                )
            # NOTE: list() converts 2D+ array into per-row sub-arrays, needed because pandas rejects >1D per-column arrays
            signals[col] = list(value) if value.ndim > 1 else value
        return signals
