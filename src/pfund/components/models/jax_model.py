from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from jax import Array
    from narwhals.typing import IntoDataFrame

import os
import pickle

import pandas as pd
import polars as pl
from pfund_kit.logging.filters.trimmed_path_filter import TrimmedPathFilter

from pfund.components.models.model_base import BaseModel

trim_path = TrimmedPathFilter.trim_path


class JaxModel(BaseModel):
    # For Flax, user assigns params separately; for Equinox, params live inside self.model.
    params: Any = None

    def _is_equinox(self) -> bool:
        try:
            import equinox as eqx
        except ImportError:
            return False
        return isinstance(self.model, eqx.Module)

    def load(self) -> dict:
        file_path = self._get_file_path(extension=".jax")
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                obj = pickle.load(f)
            if self._is_equinox():
                self.model = obj["model"]
            else:
                self.params = obj["params"]
            self._assert_no_missing_datas(obj)
            self.logger.debug(
                f"loaded trained '{self.name}' from {trim_path(file_path)}"
            )
            return obj
        return {}

    def dump(self, obj: dict[str, Any] | None = None):
        if obj is None:
            obj = {}
        if self._is_equinox():
            obj["model"] = self.model
        else:
            obj["params"] = self.params
        obj["datas"] = self.datas
        file_path = self._get_file_path(extension=".jax")
        with open(file_path, "wb") as f:
            pickle.dump(obj, f)
        self.logger.debug(f"dumped trained '{self.name}' to {trim_path(file_path)}")

    def predict(self, X: Array | IntoDataFrame, *args, **kwargs) -> Array:
        import jax.numpy as jnp

        if isinstance(X, (pd.DataFrame, pl.DataFrame)):
            X = jnp.asarray(X.to_numpy())
        elif isinstance(X, pl.LazyFrame):
            X = jnp.asarray(X.collect().to_numpy())
        elif not isinstance(X, jnp.ndarray):
            raise ValueError(f"Unsupported data type: {type(X)}")

        if self._is_equinox():
            pred_y = self.model(X, *args, **kwargs)
        else:
            if self.params is None:
                raise ValueError(
                    f"'{self.name}' is using Flax but self.params is not set"
                )
            pred_y = self.model.apply(self.params, X, *args, **kwargs)

        if not self._signal_cols:
            num_cols = pred_y.shape[-1]
            signal_cols = self._get_default_signal_cols(num_cols)
            self.set_signal_cols(signal_cols)
        return pred_y
