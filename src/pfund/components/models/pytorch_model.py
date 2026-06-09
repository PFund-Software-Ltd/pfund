from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from narwhals.typing import IntoDataFrame
    from torch import Tensor

import os

import pandas as pd
import polars as pl
from pfund_kit.logging.filters.trimmed_path_filter import TrimmedPathFilter

from pfund.components.models.model_base import BaseModel

trim_path = TrimmedPathFilter.trim_path


class PytorchModel(BaseModel):
    def load(self) -> dict:
        import torch

        file_path = self._get_file_path(extension=".pt")
        if os.path.exists(file_path):
            obj = torch.load(file_path)
            self.model.load_state_dict(obj["state_dict"])
            self._assert_no_missing_datas(obj)
            self.logger.debug(
                f"loaded trained '{self.name}' from {trim_path(file_path)}"
            )
            return obj
        return {}

    def dump(self, obj: dict[str, any] | None = None) -> bytes:
        from safetensors.torch import save

        # TODO: move these fields to metadata
        # TODO, refer to model_base, e.g. dump self.datas
        # 'dataset_periods': {
        #     'train_period': "2020-01-01 to 2020-12-31",
        #     'dev_period': ...,
        #     'test_period': ...,
        # },
        # "data_info": {
        #     'data_source': ...,
        #     'resolution': ...,
        # }
        # TODO: need to dump datasets (parquet.gz) as well?
        # return bytes, not a file — pfeed's BlobIO owns persistence (writes the .pth).
        # The component only knows its own format.
        data = save(self.model.state_dict())
        self.logger.debug(f"dumped trained '{self.name}' to bytes")
        return data

    # TODO:
    def checkpoint(self):
        pass

    def predict(self, X: Tensor | IntoDataFrame, *args, **kwargs) -> Tensor:
        import torch

        if isinstance(X, (pd.DataFrame, pl.DataFrame)):
            X = torch.tensor(X.to_numpy(), dtype=torch.float32)
        elif isinstance(X, pl.LazyFrame):
            X = torch.tensor(X.collect().to_numpy(), dtype=torch.float32)
        elif not isinstance(X, torch.Tensor):
            raise ValueError(f"Unsupported data type: {type(X)}")
        pred_y = self.model(X, *args, **kwargs)

        if not self._signal_cols:
            num_cols = pred_y.shape[-1]
            signal_cols = self._get_default_signal_cols(num_cols)
            self.set_signal_cols(signal_cols)
        return pred_y


PyTorchModel = PytorchModel
