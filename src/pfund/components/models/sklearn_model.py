from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, cast

if TYPE_CHECKING:
    import polars as pl
    from numpy.typing import NDArray
    from sklearn.base import BaseEstimator

    class _Predictor(Protocol):
        def predict(self, X: Any, *args: Any, **kwargs: Any) -> NDArray[Any]: ...

import io

import joblib

from pfund.components.models.model_base import BaseModel


class SKLearnModel[UnderlyingModelT: BaseEstimator = BaseEstimator](
    BaseModel[UnderlyingModelT]
):
    _artifact_extension = ".joblib"

    def predict(self, X: pl.DataFrame, *args: Any, **kwargs: Any) -> NDArray[Any]:
        # BaseEstimator doesn't declare predict(), every fitted sklearn estimator does
        pred_y = cast("_Predictor", self.model).predict(X, *args, **kwargs)
        return pred_y

    def _encode(self) -> bytes:
        buffer = io.BytesIO()
        joblib.dump(self.model, buffer)
        return buffer.getvalue()

    def _decode(self, data: bytes) -> None:
        self.model = joblib.load(io.BytesIO(data))
