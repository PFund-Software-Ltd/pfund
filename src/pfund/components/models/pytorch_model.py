# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false, reportUnknownVariableType=false, reportAssignmentType=false, reportArgumentType=false, reportCallIssue=false
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor
    from narwhals.typing import IntoDataFrame

import io
import json

import torch
import numpy as np
import narwhals as nw
from safetensors.torch import save, load

from pfund.components.models.model_base import BaseModel


class PyTorchModel(BaseModel):
    def predict(self, X: IntoDataFrame, *args: Any, **kwargs: Any) -> Tensor:
        # inference: eval() gives dropout/batchnorm their inference behavior and
        # no_grad() skips the autograd graph. restore the prior mode so calling
        # predict() doesn't silently leave a still-training model stuck in eval.
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                pred_y = self.model(self._to_tensor(X), *args, **kwargs)
        finally:
            self.model.train(was_training)
        return pred_y

    def _to_tensor(self, X: Any) -> Tensor:
        """Coerce X to a tensor matching the model's dtype and device.

        Accepts a torch.Tensor (passed through), a numpy array, or any
        dataframe — native (pandas/polars/pyarrow/...) or an already-wrapped
        narwhals frame, eager or lazy.
        """
        # align the input with the model's device AND dtype (e.g. after
        # model.to("cuda") or a float64/bf16 model), else self.model(X) raises on
        # a device or dtype mismatch in the matmul; parameterless modules have no
        # parameter to read, so fall back to cpu/float32
        try:
            param = next(self.model.parameters())
            device, dtype = param.device, param.dtype
        except StopIteration:
            device, dtype = torch.device("cpu"), torch.float32

        if isinstance(X, torch.Tensor):
            return X.to(device=device, dtype=dtype)
        if isinstance(X, np.ndarray):
            array = X
        else:
            # accept native frames (pandas/polars/pyarrow/...) or already-wrapped
            # narwhals frames; pass_through returns non-frames unchanged so we can
            # reject them with a clear error instead of a cryptic narwhals one
            frame = (
                X
                if isinstance(X, (nw.DataFrame, nw.LazyFrame))
                else nw.from_native(X, pass_through=True)
            )
            if not isinstance(frame, (nw.DataFrame, nw.LazyFrame)):
                raise TypeError(
                    f"unsupported data type for '{X.__class__.__name__}'; "
                    + "predict() accepts a torch.Tensor, numpy array, or dataframe"
                )
            if isinstance(frame, nw.LazyFrame):
                frame = frame.collect()
            array = frame.to_numpy()
        return torch.tensor(array, dtype=dtype, device=device)

    def _encode(self, payload: dict[str, Any]) -> bytes:
        model = payload["model"]
        metadata = {"signal_cols": json.dumps(payload["signal_cols"])}
        data = save(model.state_dict(), metadata=metadata)
        return data

    def _decode(self, data: bytes) -> None:
        # frame torch's raw strict-load error so architecture drift reads as a
        # pfund message instead of a bare "Missing key(s) in state_dict"
        try:
            self.model.load_state_dict(load(data))
        except RuntimeError as err:
            raise RuntimeError(
                f"failed to load saved weights into '{self.name}': the saved "
                + "state_dict does not match the current model architecture"
            ) from err
        self.set_signal_cols(self._read_safetensors_signal_cols(data))

    def _encode_checkpoint(self, checkpoint: dict[str, Any]) -> bytes:
        # user owns the checkpoint dict; pfund only stamps in the framework
        # essential (signal_cols) so the wrapper restores on resume.
        buffer = io.BytesIO()
        torch.save({**checkpoint, "signal_cols": self._signal_cols}, buffer)
        return buffer.getvalue()

    def _decode_checkpoint(self, data: bytes) -> dict[str, Any]:
        # weights_only=False: it's a user dict, not a bare state_dict, and pfund wrote it.
        # map_location="cpu": land every tensor (incl. optimizer state) on CPU so a
        # GPU-saved checkpoint reloads on any box; the caller re-homes with .to(device).
        checkpoint = torch.load(
            io.BytesIO(data), weights_only=False, map_location="cpu"
        )
        self.set_signal_cols(checkpoint.pop("signal_cols"))
        return checkpoint
