from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor

import io

import numpy as np
import polars as pl
import safetensors.torch
import torch
import torch.nn as nn

from pfund.components.bar_dataframe import KEY_COLS as BAR_KEY_COLS
from pfund.components.models.model_base import BaseModel


class PyTorchModel[UnderlyingModelT: nn.Module = nn.Module](
    BaseModel[UnderlyingModelT]
):
    _artifact_extension = ".safetensors"
    _checkpoint_extension = ".pth"

    def predict(self, X: pl.DataFrame, *args: Any, **kwargs: Any) -> Tensor:
        # inference: eval() gives dropout/batchnorm their inference behavior and
        # no_grad() skips the autograd graph. restore the prior mode so calling
        # predict() doesn't silently leave a still-training model stuck in eval.
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                pred_y = self.model(self.to_tensor(X), *args, **kwargs)
        finally:
            self.model.train(was_training)
        return pred_y

    def to_tensor(self, X: Tensor | np.ndarray | pl.DataFrame) -> Tensor:
        """Coerce X to a tensor matching the model's dtype and device.

        Accepts a torch.Tensor (passed through), a numpy array, or a polars dataframe.
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
        elif isinstance(X, pl.DataFrame):
            # PFund keeps bar keys in feature dataframes for row alignment, but
            # date/product/resolution are not numerical model inputs.
            key_cols = [col for col in BAR_KEY_COLS if col in X.columns]
            array = X.drop(key_cols).to_numpy()
        else:
            raise TypeError(
                f"unsupported data type for '{X.__class__.__name__}'; "
                + "predict() accepts a torch.Tensor, numpy array, or polars dataframe"
            )
        return torch.tensor(array, dtype=dtype, device=device)

    def _encode(self) -> bytes:
        return safetensors.torch.save(self.model.state_dict())

    def _decode(self, data: bytes) -> None:
        # frame torch's raw strict-load error so architecture drift reads as a
        # pfund message instead of a bare "Missing key(s) in state_dict"
        try:
            self.model.load_state_dict(safetensors.torch.load(data))
        except RuntimeError as err:
            raise RuntimeError(
                f"failed to load saved weights into '{self.name}': the saved "
                + "state_dict does not match the current model architecture"
            ) from err

    def _encode_checkpoint(self, checkpoint: dict[str, Any]) -> bytes:
        # the user owns the checkpoint dict, pfund saves it as is
        buffer = io.BytesIO()
        torch.save(checkpoint, buffer)
        return buffer.getvalue()

    def _decode_checkpoint(self, data: bytes) -> dict[str, Any]:
        # weights_only=False: it's a user dict, not a bare state_dict, and pfund wrote it.
        # map_location="cpu": land every tensor (incl. optimizer state) on CPU so a
        # GPU-saved checkpoint reloads on any box; the caller re-homes with .to(device).
        return torch.load(io.BytesIO(data), weights_only=False, map_location="cpu")
