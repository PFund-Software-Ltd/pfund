# TODO: not tested/officially supported
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from jax import Array

import io
import pickle

import numpy as np
import polars as pl
import jax
import jax.numpy as jnp

from pfund.components.bar_dataframe import KEY_COLS as BAR_KEY_COLS
from pfund.components.models.model_base import BaseModel


class JAXModel(BaseModel):
    """Wraps a Flax or Equinox model.

    Flax: assign the model's variables to `self.variables` — this is the FULL
    variables collection, not just the "params" subtree. For models with
    BatchNorm (or any mutable collection) it must include those too, e.g.
    `self.variables = {"params": ..., "batch_stats": ...}`; otherwise apply()
    raises ScopeCollectionNotFound. save()/load() persist every collection.

    (`self.variables`, not `self.params`: `params` is reserved on every
    component for user-declared hyperparameters — see ComponentMixin.params.)

    Inference determinism is the caller's responsibility: JAX has no global
    eval() switch, so dropout/batchnorm behaviour is controlled per call via
    the model's own apply kwargs (e.g. train=False / deterministic=True / an
    rngs= key), forwarded through predict(*args, **kwargs).

    Equinox: params live inside `self.model` (a PyTree). save()/load()
    serialise the model's array leaves via eqx; load() fills them into the
    current `self.model`, so its architecture must already match (like
    torch's load_state_dict).
    """

    variables: Any = None

    def predict(self, X: pl.DataFrame, *args: Any, **kwargs: Any) -> Array:
        X_array = self._to_array(X)
        if self._is_equinox():
            pred_y = self.model(X_array, *args, **kwargs)
        else:
            if self.variables is None:
                raise ValueError(
                    f"'{self.name}' is using Flax but self.variables is not set"
                )
            pred_y = self.model.apply(self.variables, X_array, *args, **kwargs)
        return pred_y

    @staticmethod
    def _to_array(X: Array | np.ndarray | pl.DataFrame) -> Array:
        """Coerce X to a jax array.

        Accepts a jax array (passed through), a numpy array, or a polars dataframe.
        """
        if isinstance(X, jax.Array):
            return X
        if isinstance(X, np.ndarray):
            return jnp.asarray(X)
        if not isinstance(X, pl.DataFrame):
            raise TypeError(
                f"unsupported data type for '{X.__class__.__name__}'; "
                + "predict() accepts a jax array, numpy array, or polars dataframe"
            )
        # PFund keeps bar keys in feature dataframes for row alignment, but
        # date/product/resolution are not numerical model inputs.
        key_cols = [col for col in BAR_KEY_COLS if col in X.columns]
        return jnp.asarray(X.drop(key_cols).to_numpy())

    def _is_equinox(self) -> bool:
        try:
            import equinox as eqx
        except ImportError:
            return False
        return isinstance(self.model, eqx.Module)

    def _encode(self) -> bytes:
        if self._is_equinox():
            import equinox as eqx

            buffer = io.BytesIO()
            eqx.tree_serialise_leaves(buffer, self.model)
            return buffer.getvalue()

        from flax.traverse_util import flatten_dict
        from safetensors.flax import save

        if self.variables is None:
            raise ValueError(
                f"'{self.name}' is using Flax but self.variables is not set"
            )

        flat_params = {
            "/".join(key): value for key, value in flatten_dict(self.variables).items()
        }
        return save(flat_params)

    def _decode(self, data: bytes) -> None:
        if self._is_equinox():
            import equinox as eqx

            # deserialise leaves INTO the current model as the structural template —
            # like load_state_dict, self.model must already be the right architecture
            self.model = eqx.tree_deserialise_leaves(io.BytesIO(data), self.model)
            return

        from flax.traverse_util import unflatten_dict
        from safetensors.flax import load

        self.variables = unflatten_dict(
            {tuple(k.split("/")): v for k, v in load(data).items()}
        )

    # --- checkpointing (training state: params + optimizer + bookkeeping) --------
    # The checkpoint is the user's opaque training-state pytree, so there's no
    # equinox/flax branch here. pickle (not orbax/msgpack): it's a single blob like
    # the torch path and preserves node types, so optax opt_state NamedTuples
    # round-trip intact and optimizer.update() resumes — a target-less orbax/msgpack
    # restore would flatten them to dicts and break resume.

    def _encode_checkpoint(self, checkpoint: dict[str, Any]) -> bytes:
        # the user owns the checkpoint pytree, pfund saves it as is
        return pickle.dumps(checkpoint)

    def _decode_checkpoint(self, data: bytes) -> dict[str, Any]:
        # pfund wrote this blob; pickle rebuilds the pytree's node types as-is, so
        # optax opt_state NamedTuples survive and optimizer.update() can resume.
        return pickle.loads(data)
