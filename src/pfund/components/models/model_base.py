from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, override

if TYPE_CHECKING:
    import equinox as eqx
    import polars as pl
    import torch.nn as nn
    from flax import linen
    from jax import Array
    from numpy.typing import NDArray
    from sklearn.base import BaseEstimator
    from torch import Tensor

    from pfund._pfeed_source.component.data_handler import TrainedOn
    from pfund.typing import Signals

    type UnderlyingModel = nn.Module | BaseEstimator | linen.Module | eqx.Module

import sys
from abc import ABC, abstractmethod

import numpy as np

from pfund.components._mixins.component_mixin import ComponentMixin
from pfund.components.models.model_meta import MetaModel
from pfund.enums import ArtifactType, ComponentType


def _get_caller_source_file_path() -> str | None:
    """Return the first caller source file outside pfund itself."""
    import inspect
    from pathlib import Path

    pfund_package_path = Path(__file__).resolve().parents[2]
    frame = inspect.currentframe()
    try:
        frame = frame.f_back if frame is not None else None
        while frame is not None:
            source_file_path = Path(frame.f_code.co_filename).resolve()
            if (
                not source_file_path.is_relative_to(pfund_package_path)
                and source_file_path.is_file()
            ):
                return str(source_file_path)
            frame = frame.f_back
    finally:
        del frame
    return None


class BaseModel[UnderlyingModelT: UnderlyingModel = Any](
    ComponentMixin, ABC, metaclass=MetaModel
):
    """Wraps an underlying ML model, e.g. a torch nn.Module or an sklearn estimator.

    Generic over the underlying model's type, so `self.model` keeps it,
    e.g. self.model is TorchLinearRegression in PyTorchModel[TorchLinearRegression].

    The default is Any, not UnderlyingModel: BaseModel is invariant in it (self.model is writable),
    so a bare BaseModel annotation, e.g. a [ModelT: BaseModel] bound, must accept any BaseModel[...].
    """

    _allowed_model_method_collisions: ClassVar[frozenset[str]] = frozenset({"predict"})
    # file extensions of the bytes _encode() / _encode_checkpoint() produce, e.g. model.safetensors in storage.
    # adapters set their format's extension, a user model subclassing BaseModel directly can override them
    _artifact_extension: ClassVar[str] = ".bin"
    _checkpoint_extension: ClassVar[str] = ".bin"

    def __init__(self, model: UnderlyingModelT, *args: Any, **kwargs: Any):
        self.__mixin_post_init__(ComponentType.model, model, *args, **kwargs)
        self.model: UnderlyingModelT = model
        # Built-in adapters otherwise resolve their source artifact to this
        # pfund file. Capture the user's construction site here so both
        # PyTorchModel(model) and automatic wrapping retain user provenance.
        # User-defined wrapper classes keep their class file as the artifact.
        if type(self).__module__.startswith("pfund.") and (
            source_file_path := _get_caller_source_file_path()
        ):
            self._set_source_file_path(source_file_path)
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
        self, X: pl.DataFrame, *args: Any, **kwargs: Any
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
            # NOT self.name: _name may be unset too (e.g. mid-unpickle), and self.name would recurse back here
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
        try:
            return getattr(model, name)
        except AttributeError as err:
            # from err, not from None: the AttributeError may come from inside a property of the model
            raise AttributeError(
                f"'{self.name}' and its underlying model '{self.model.__class__.__name__}' both have no attribute '{name}'"
            ) from err

    def to_signals(self, pred: NDArray[Any] | Tensor | Array) -> Signals:
        """Converts the output of predict() to signals

        Args:
            pred: the prediction returned by predict()

        Returns:
            dict[ColumnName, Any]: The predicted signals.
        """
        torch = sys.modules.get("torch")
        is_tensor = torch is not None and isinstance(pred, torch.Tensor)
        # np.asarray() normalizes JAX/other array-likes; no copy if already ndarray
        pred = pred.detach().cpu().numpy() if is_tensor else np.asarray(pred)
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

    @override
    def _get_source_classes(self) -> list[type]:
        # the underlying model is code too, e.g. the user's nn.Module wrapped by pf.PyTorchModel
        return [*super()._get_source_classes(), *type(self.model).__mro__]

    @override
    def _materialize(self):
        self.load()
        super()._materialize()

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "model": self.model.__class__.__name__,
        }

    @abstractmethod
    def _encode(self) -> bytes: ...

    @abstractmethod
    def _decode(self, data: bytes) -> None: ...

    def load(self) -> None:
        data, metadata = self._feed.load_artifact(
            self, ArtifactType.model, include_metadata=True
        )
        if data is None:
            raise FileNotFoundError(
                f"no saved model found for '{self.name}'; call save() after training first"
            )
        if not isinstance(data, bytes):
            raise TypeError(
                f"saved model for '{self.name}' returned {type(data).__name__}, expected bytes"
            )
        # warn only: the trained weights are still valid data, the user decides whether to retrain
        if reason := self._get_stale_reason(metadata):
            self.logger.warning(
                f"'{self.name}' saved model is stale: {reason}; "
                + "retrain and save() it if the change affects the model"
            )
        self._decode(data)

    def _get_trained_on(self) -> TrainedOn | None:
        """What the model was trained on, stamped on its model and checkpoint artifacts.

        The whole data range, overridden in backtesting, where the engine can split it.
        None if the engine has no data range, e.g. live trading.
        """
        from pfund._pfeed_source.component.data_handler import TrainedOn

        data_start, data_end = self.context.data_start, self.context.data_end
        if data_start is None or data_end is None:
            return None
        return TrainedOn.whole_range(data_start, data_end)

    def save(self) -> None:
        artifact = self._encode()
        self._feed.save_artifact(self, ArtifactType.model, artifact=artifact)
