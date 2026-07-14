# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false, reportUnknownVariableType=false, reportAssignmentType=false, reportArgumentType=false, reportCallIssue=false
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from narwhals.typing import IntoDataFrame

import io
import joblib

from pfund.components.models.model_base import BaseModel


class SKLearnModel(BaseModel):
    def predict(self, X: IntoDataFrame, *args: Any, **kwargs: Any) -> NDArray[Any]:
        pred_y = self.model.predict(X, *args, **kwargs)
        return pred_y

    def _encode(self, payload: dict[str, Any]) -> bytes:
        buffer = io.BytesIO()
        joblib.dump(payload, buffer)
        return buffer.getvalue()

    def _decode(self, data: bytes) -> None:
        payload = joblib.load(io.BytesIO(data))
        self.model = payload["model"]
        self.set_signal_cols(payload["signal_cols"])
