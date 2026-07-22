# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false, reportUnknownVariableType=false, reportAssignmentType=false, reportArgumentType=false, reportCallIssue=false
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from jax import Array
    from narwhals.typing import IntoDataFrame

import io
import json
import pickle
import struct

import numpy as np
import narwhals as nw
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

    def predict(self, X: IntoDataFrame, *args: Any, **kwargs: Any) -> Array:
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
    def _to_array(X: Any) -> Array:
        """Coerce X to a jax array.

        Accepts a jax array (passed through), a numpy array, or any dataframe —
        native (pandas/polars/pyarrow/...) or an already-wrapped narwhals frame,
        eager or lazy.
        """
        if isinstance(X, jax.Array):
            return X
        if isinstance(X, np.ndarray):
            return jnp.asarray(X)
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
                + "predict() accepts a jax array, numpy array, or dataframe"
            )
        if isinstance(frame, nw.LazyFrame):
            frame = frame.collect()
        # PFund keeps bar keys in feature dataframes for row alignment, but
        # date/product/resolution are not numerical model inputs.
        key_cols = [col for col in BAR_KEY_COLS if col in frame.columns]
        if key_cols:
            frame = frame.drop(key_cols)
        return jnp.asarray(frame.to_numpy())

    def _is_equinox(self) -> bool:
        try:
            import equinox as eqx
        except ImportError:
            return False
        return isinstance(self.model, eqx.Module)

    def _encode(self, payload: dict[str, Any]) -> bytes:
        if self._is_equinox():
            import equinox as eqx

            # eqx serialises only array leaves and has no metadata slot, so wrap it:
            #   [8-byte LE length][signal_cols json][eqx leaf bytes]
            signal_cols = json.dumps(payload["signal_cols"]).encode()
            buffer = io.BytesIO()
            eqx.tree_serialise_leaves(buffer, self.model)
            return struct.pack("<Q", len(signal_cols)) + signal_cols + buffer.getvalue()

        from flax.traverse_util import flatten_dict
        from safetensors.flax import save

        if self.variables is None:
            raise ValueError(
                f"'{self.name}' is using Flax but self.variables is not set"
            )

        flat_params = {
            "/".join(key): value for key, value in flatten_dict(self.variables).items()
        }
        metadata = {"signal_cols": json.dumps(payload["signal_cols"])}
        data = save(flat_params, metadata=metadata)
        return data

    def _decode(self, data: bytes) -> None:
        if self._is_equinox():
            import equinox as eqx

            (n,) = struct.unpack("<Q", data[:8])
            signal_cols = json.loads(data[8 : 8 + n])
            # deserialise leaves INTO the current model as the structural template —
            # like load_state_dict, self.model must already be the right architecture
            self.model = eqx.tree_deserialise_leaves(
                io.BytesIO(data[8 + n :]), self.model
            )
            self.set_signal_cols(signal_cols)
            return

        from flax.traverse_util import unflatten_dict
        from safetensors.flax import load

        self.variables = unflatten_dict(
            {tuple(k.split("/")): v for k, v in load(data).items()}
        )
        self.set_signal_cols(self._read_safetensors_signal_cols(data))

    # --- checkpointing (training state: params + optimizer + bookkeeping) --------
    # The checkpoint is the user's opaque training-state pytree, so there's no
    # equinox/flax branch here. pickle (not orbax/msgpack): it's a single blob like
    # the torch path and preserves node types, so optax opt_state NamedTuples
    # round-trip intact and optimizer.update() resumes — a target-less orbax/msgpack
    # restore would flatten them to dicts and break resume.

    def _encode_checkpoint(self, checkpoint: dict[str, Any]) -> bytes:
        # user owns the checkpoint pytree; pfund only stamps in the framework
        # essential (signal_cols) so the wrapper restores on resume.
        return pickle.dumps({**checkpoint, "signal_cols": self._signal_cols})

    def _decode_checkpoint(self, data: bytes) -> dict[str, Any]:
        # pfund wrote this blob; pickle rebuilds the pytree's node types as-is, so
        # optax opt_state NamedTuples survive and optimizer.update() can resume.
        checkpoint = pickle.loads(data)
        self.set_signal_cols(checkpoint.pop("signal_cols"))
        return checkpoint
