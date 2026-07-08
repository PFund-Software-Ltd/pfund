from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    import torch.nn as nn
    from sklearn.base import BaseEstimator

    from pfund.components.actor_proxy import ActorProxy
    from pfund.components.models.model_base import BaseModel
    from pfund.typing import ModelT


__all__ = ["wrap_model"]


def wrap_model(
    model: BaseEstimator | nn.Module | ModelT | ActorProxy[ModelT],
) -> ModelT | ActorProxy[ModelT]:
    """Wraps a sklearn/pytorch/jax model into a pfund model"""
    from pfund.components.actor_proxy import ActorProxy
    from pfund.components.models.model_base import BaseModel

    if isinstance(model, (ActorProxy, BaseModel)):
        return model
    for wrap in (_wrap_sklearn, _wrap_pytorch, _wrap_jax):
        if (pfund_model := wrap(model)) is not None:
            pfund_model._set_name(type(model).__name__)
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
