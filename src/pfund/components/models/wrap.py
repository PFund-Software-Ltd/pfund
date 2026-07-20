from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from pfund.components.actor_proxy import ActorProxy
    from pfund.components.models.model_base import BaseModel, UnderlyingModel
    from pfund.typing import ModelT


__all__ = ["wrap_model"]


def _get_caller_source_artifact_path() -> str | None:
    """Return the first caller source file outside pfund itself."""
    import inspect
    from pathlib import Path

    pfund_package_path = Path(__file__).resolve().parents[2]
    frame = inspect.currentframe()
    try:
        frame = frame.f_back if frame is not None else None
        while frame is not None:
            source_artifact_path = Path(frame.f_code.co_filename).resolve()
            if (
                not source_artifact_path.is_relative_to(pfund_package_path)
                and source_artifact_path.is_file()
            ):
                return str(source_artifact_path)
            frame = frame.f_back
    finally:
        del frame
    return None


def wrap_model(
    model: UnderlyingModel | ModelT | ActorProxy[ModelT],
) -> ModelT | ActorProxy[ModelT]:
    """Wraps a sklearn/pytorch/jax model into a pfund model"""
    from pfund.components.actor_proxy import ActorProxy
    from pfund.components.models.model_base import BaseModel

    if isinstance(model, (ActorProxy, BaseModel)):
        return model
    for wrap in (_wrap_sklearn, _wrap_pytorch, _wrap_jax):
        if (pfund_model := wrap(model)) is not None:
            pfund_model._set_name(type(model).__name__)
            # A raw model does not retain the file where it was instantiated,
            # while its wrapper class points to pfund's internal adapter. Capture
            # the user's call site before that source provenance is lost.
            if source_artifact_path := _get_caller_source_artifact_path():
                pfund_model._set_source_artifact_path(source_artifact_path)
            return cast("ModelT", pfund_model)
    raise TypeError(f"Unsupported model type: {type(model).__name__}")


def _wrap_sklearn(model: object) -> BaseModel | None:
    try:
        from sklearn.base import BaseEstimator
    except ImportError:
        return None
    if not isinstance(model, BaseEstimator):
        return None
    from pfund.components.models.sklearn_model import SKLearnModel

    return SKLearnModel(model)


def _wrap_pytorch(model: object) -> BaseModel | None:
    try:
        import torch.nn as nn
    except ImportError:
        return None
    if not isinstance(model, nn.Module):
        return None
    from pfund.components.models.pytorch_model import PyTorchModel

    return PyTorchModel(model)


def _wrap_jax(model: object) -> BaseModel | None:
    # jax comes in two flavors JAXModel handles: Flax (linen.Module) and Equinox (eqx.Module)
    if not _is_jax_model(model):
        return None
    from pfund.components.models.jax_model import JAXModel

    return JAXModel(model)


def _is_jax_model(model: object) -> bool:
    try:
        from flax import linen

        if isinstance(model, linen.Module):
            return True
    except ImportError:
        pass
    try:
        import equinox as eqx

        if isinstance(model, eqx.Module):
            return True
    except ImportError:
        pass
    return False
