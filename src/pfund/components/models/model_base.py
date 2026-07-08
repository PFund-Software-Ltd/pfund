# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false, reportUnknownVariableType=false, reportAssignmentType=false, reportArgumentType=false, reportUnknownParameterType=false
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import torch.nn as nn
    from narwhals.typing import IntoDataFrame
    from numpy import ndarray
    from sklearn.base import BaseEstimator
    from torch import Tensor
    from pfeed.dataflow.result import RunResult

    from pfund.components.indicators.indicator_base import TalibFunction
    from pfund.typing import ColumnName

    MachineLearningModel = nn.Module | BaseEstimator | TalibFunction

from abc import ABC, abstractmethod

import narwhals as nw

from pfund.enums import ArtifactType
from pfund.components.mixin import ComponentMixin
from pfund.components.models.model_meta import MetaModel


class BaseModel(ComponentMixin, ABC, metaclass=MetaModel):
    def __init__(self, model: MachineLearningModel, *args: Any, **kwargs: Any):
        self.model = model  # user-defined machine learning model
        self._df_form: Literal["wide", "long"] = "wide"
        self.__mixin_post_init__(
            model, *args, **kwargs
        )  # calls ComponentMixin.__mixin_post_init__()

    @abstractmethod
    def predict(self, X: Any, *args: Any, **kwargs: Any) -> Any:
        pass

    def __getattr__(self, name: str):
        """
        __getattr__ only fires when the attribute isn't found via normal lookup,
        so we only reach here for methods/attrs not defined on the wrapper.
        """
        # Skip underscore names:
        # (1) dunders like __reduce__/__getstate__ have
        # pickle-protocol meaning and must not silently resolve to self.model,
        # (2) single-underscore names are pfund-internal state — delegating them
        # risks silent collisions if the underlying model grows a same-named attr,
        # (3) avoids recursion during __init__/unpickle when private attrs may be
        # accessed before they're set on self.
        if name.startswith("_"):
            raise AttributeError(name)
        model = self.__dict__.get("model")
        if model is None:
            raise AttributeError(f"'{self.name}' has no attribute '{name}'")
        try:
            return getattr(model, name)
        except AttributeError:
            raise AttributeError(
                f"'{self.name}' and its underlying model '{self.model.__class__.__name__}' both have no attribute '{name}'"
            )

    def signalize(self, features_df: IntoDataFrame) -> dict[ColumnName, Any]:
        """Creates signals_df (combined signals from other component)
        Args:
            data_df: dataframe in {self._df_form} form
        """
        X = nw.from_native(features_df)
        pred: Tensor | ndarray = self.predict(X)
        is_from_pytorch = type(pred).__module__.startswith("torch")
        if is_from_pytorch:
            pred = pred.detach().cpu().numpy()
        signal_cols = self._signal_cols
        num_signal_cols = len(signal_cols)
        if pred.ndim == 1:
            if num_signal_cols != 1:
                raise ValueError(
                    f"prediction is 1D but {self.name} has {num_signal_cols} signal columns: {signal_cols}"
                )
            signals_dict = {signal_cols[0]: pred}
        else:
            # last dimension = signal columns, everything in between is packed into cells
            if num_signal_cols != pred.shape[-1]:
                raise ValueError(
                    f"Expected {num_signal_cols} signal columns for {self.name}, but prediction has shape {pred.shape}"
                )
            signals_dict = {}
            for i, col in enumerate(signal_cols):
                values = pred[..., i]
                # NOTE: list() converts 2D+ array into per-row sub-arrays, needed because pandas rejects >1D per-column arrays
                signals_dict[col] = list(values) if values.ndim > 1 else values
        return signals_dict

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "model": self.model.__class__.__name__,
        }

    def _assert_functions_signatures(self):
        from pfund_kit.utils.function import get_function_args_and_kwargs

        super()._assert_functions_signatures()

        def _assert_predict_function():
            args, kwargs, _, _ = get_function_args_and_kwargs(self.predict)
            if not args or args[0] != "X":
                raise Exception(
                    f'{self.name} predict() must have "X" as its first arg, i.e. predict(self, X, *args, **kwargs)'
                )

        _assert_predict_function()

    def _get_default_signal_cols(self, num_cols: int) -> list[str]:
        if num_cols == 1:
            columns = [self.name]
        else:
            columns = [f"{self.name}-{i}" for i in range(num_cols)]
        return columns

    @abstractmethod
    def _encode(self, payload: dict[str, Any]) -> bytes: ...

    @abstractmethod
    def _decode(self, data: bytes) -> None: ...

    @staticmethod
    def _read_safetensors_signal_cols(data: bytes) -> list[str]:
        # safetensors layout: 8-byte LE u64 header length, then a JSON header
        # whose "__metadata__" holds the str->str dict _encode wrote signal_cols into.
        import json
        import struct

        (header_len,) = struct.unpack("<Q", data[:8])
        header = json.loads(data[8 : 8 + header_len])
        signal_cols = header.get("__metadata__", {}).get("signal_cols")
        return json.loads(signal_cols) if signal_cols else []

    def load(self) -> None:
        data: bytes = self.store._feed.retrieve(
            artifact_type=ArtifactType.model,
            storage_config=self.store.storage_config,
        ).run()
        if data:
            self._decode(data)
        else:
            raise FileNotFoundError(
                f"no saved model found for '{self.name}' in "
                + f"{self.store.storage_config.storage} storage; call save() after training first"
            )

    def save(self) -> RunResult:
        return self.store._feed.download(
            artifact_type=ArtifactType.model,
            storage_config=self.store.storage_config,
        ).run()

    def serialize(self) -> bytes:
        payload = {"model": self.model, "signal_cols": self._signal_cols}
        return self._encode(payload)

    def load_checkpoint(self, step: int) -> dict[str, Any]:
        """Read back the checkpoint written by save_checkpoint() at `step`. Restores
        the framework essentials (signal_cols) into the model and returns the user's
        own dict (weights, optimizer, epoch, ...) so they can resume their loop.
        """
        if not hasattr(self, "_decode_checkpoint"):
            raise NotImplementedError(
                "load_checkpoint is not implemented for this model"
            )
        data: bytes = self.store._feed.retrieve(
            artifact_type=ArtifactType.model,
            storage_config=self.store.storage_config,
            checkpoint_step=step,
        ).run()
        if not data:
            raise FileNotFoundError(
                f"no checkpoint found for '{self.name}' at step {step} in "
                + f"{self.store.storage_config.storage} storage"
            )
        return self._decode_checkpoint(data)

    def save_checkpoint(self, checkpoint: dict[str, Any], step: int) -> RunResult:
        """Persist a training checkpoint the user built (weights, optimizer, epoch,
        ... — their dict, their content). Written under the same model storage as
        save(), but keyed by `step` so checkpoints coexist without clobbering the
        final model or each other.
        """
        if not hasattr(self, "_encode_checkpoint"):
            raise NotImplementedError(
                "save_checkpoint is not implemented for this model"
            )
        data: bytes = self._encode_checkpoint(checkpoint)
        return self.store._feed.download(
            artifact_type=ArtifactType.model,
            storage_config=self.store.storage_config,
            checkpoint_step=step,
            checkpoint_data=data,
        ).run()
