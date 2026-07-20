# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false, reportUnknownVariableType=false, reportAssignmentType=false, reportArgumentType=false, reportUnknownParameterType=false
from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    import torch.nn as nn
    import equinox as eqx
    from flax import linen
    from narwhals.typing import IntoDataFrame
    from sklearn.base import BaseEstimator
    from numpy.typing import NDArray
    from torch import Tensor
    from jax import Array
    from pfeed.dataflow.result import RunResult

    from pfund.typing import Signals

    UnderlyingModel = nn.Module | BaseEstimator | linen.Module | eqx.Module

from abc import ABC, abstractmethod

import numpy as np

from pfund.enums import ArtifactType
from pfund.components.mixin import ComponentMixin
from pfund.components.models.model_meta import MetaModel


class BaseModel(ComponentMixin, ABC, metaclass=MetaModel):
    _allowed_model_method_collisions: ClassVar[frozenset[str]] = frozenset({"predict"})

    def __init__(self, model: UnderlyingModel, *args: Any, **kwargs: Any):
        self.model: UnderlyingModel = model
        self.__mixin_post_init__(
            model, *args, **kwargs
        )  # calls ComponentMixin.__mixin_post_init__()
        self._assert_no_model_method_collisions()

    def _assert_no_model_method_collisions(self) -> None:
        import inspect

        component_methods = {
            name
            for name, _ in inspect.getmembers(type(self), predicate=callable)
            if not name.startswith("_")
        }
        underlying_model_methods = {
            name
            for name, _ in inspect.getmembers(type(self.model), predicate=callable)
            if not name.startswith("_")
        }
        collisions = (
            component_methods & underlying_model_methods
        ) - self._allowed_model_method_collisions
        if collisions:
            raise TypeError(
                f"'{self.__class__.__name__}' and its underlying model "
                + f"'{self.model.__class__.__name__}' define the same public "
                + f"method(s): {', '.join(sorted(collisions))}"
            )

    @abstractmethod
    def predict(
        self, X: IntoDataFrame, *args: Any, **kwargs: Any
    ) -> NDArray[Any] | Tensor | Array:
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

    def signalize(self, X: IntoDataFrame) -> Signals:
        """Creates signals of this component

        Args:
            X: features df

        Returns:
            dict[ColumnName, Any]: The predicted signals.
        """
        pred: Tensor | NDArray[Any] = self.predict(X)
        if pred is None:  # pyright: ignore[reportUnnecessaryComparison]
            raise ValueError(
                f"{self.name} predict() returned None, did you forget the return statement?"
            )

        is_from_pytorch = type(pred).__module__.startswith("torch")
        if is_from_pytorch:
            pred = pred.detach().cpu().numpy()
        pred = np.asarray(
            pred
        )  # normalizes JAX/other array-likes; no copy if already ndarray
        if pred.ndim == 0:
            raise ValueError(
                f"{self.name} predict() returned a scalar; "
                + "expected one value per row"
            )
        if not self._signal_cols:
            num_cols = pred.shape[-1] if pred.ndim > 1 else 1
            signal_cols = self._get_default_signal_cols(num_cols)
            self.set_signal_cols(signal_cols)

        signal_cols = self._signal_cols
        num_signal_cols = len(signal_cols)
        if pred.ndim == 1:
            if num_signal_cols != 1:
                raise ValueError(
                    f"prediction is 1D but {self.name} has {num_signal_cols} signal columns: {signal_cols}"
                )
            signals = {signal_cols[0]: pred}
        else:
            # last dimension = signal columns, everything in between is packed into cells
            if num_signal_cols != pred.shape[-1]:
                raise ValueError(
                    f"Expected {num_signal_cols} signal columns for {self.name}, but prediction has shape {pred.shape}"
                )
            signals = {}
            for i, col in enumerate(signal_cols):
                values = pred[..., i]
                # NOTE: list() converts 2D+ array into per-row sub-arrays, needed because pandas rejects >1D per-column arrays
                signals[col] = list(values) if values.ndim > 1 else values
        return signals

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "model": self.model.__class__.__name__,
        }

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

    @property
    def _model_artifact(self) -> bytes:
        return self._encode({"model": self.model, "signal_cols": self._signal_cols})

    def _materialize(self):
        super()._materialize()
        self.load()

    def load(self) -> None:
        result = self.store._feed.retrieve(
            artifact_type=ArtifactType.model,
            storage_config=self.store.storage_config,
        ).run()
        data = result.data
        if data is None:
            raise FileNotFoundError(
                f"no saved model found for '{self.name}' in "
                + f"{self.store.storage_config.storage} storage; call save() after training first"
            )
        if not isinstance(data, bytes):
            raise TypeError(
                f"saved model for '{self.name}' returned {type(data).__name__}, expected bytes"
            )
        self._decode(data)

    def save(self) -> RunResult:
        return self.store._feed.download(
            artifact_type=ArtifactType.model,
            storage_config=self.store.storage_config,
        ).run()
