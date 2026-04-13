from __future__ import annotations
from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    from typing import Protocol
    from numpy import ndarray
    class SklearnPredictor(Protocol):
        def fit(self, X: Any, y: Any, **kwargs: Any) -> Any: ...
        def predict(self, X: Any, **kwargs: Any) -> Any: ...

from pfund.components.models.model_base import BaseModel


class SklearnModel(BaseModel):
    model: SklearnPredictor

    def fit(self, X: Any, y: Any, **kwargs: Any):
        return self.model.fit(X, y, **kwargs)
    
    def predict(self, X: Any, *args: Any, **kwargs: Any) -> ndarray:
        pred_y = self.model.predict(X, *args, **kwargs)
        if not self.signal_cols:
            num_cols = pred_y.shape[-1] if pred_y.ndim > 1 else 1
            signal_cols = self._get_default_signal_cols(num_cols)
            self.set_signal_cols(signal_cols)
        return pred_y
