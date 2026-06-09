# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false, reportUnknownVariableType=false, reportAssignmentType=false, reportArgumentType=false, reportCallIssue=false
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from numpy import ndarray

from pfund.components.models.model_base import BaseModel


class SklearnModel(BaseModel):
    def predict(self, X: Any, *args: Any, **kwargs: Any) -> ndarray:
        pred_y = self.model.predict(X, *args, **kwargs)
        if not self._signal_cols:
            num_cols = pred_y.shape[-1] if pred_y.ndim > 1 else 1
            signal_cols = self._get_default_signal_cols(num_cols)
            self.set_signal_cols(signal_cols)
        return pred_y

    def dump(self, obj: dict[str, Any] | None = None) -> bytes:
        import io

        import joblib

        if obj is None:
            obj = {}
        obj.update(
            {
                "model": self.model,
                "datas": self.datas,
            }
        )
        # serialize to bytes, not a file — pfeed's BlobIO owns persistence (writes
        # the .joblib). The component only knows its own format.
        buffer = io.BytesIO()
        joblib.dump(obj, buffer)
        return buffer.getvalue()


SKLearnModel = SklearnModel
