from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import pandas as pd
    import torch

import os

try:
    import polars as pl
except ImportError:
    pl = None

from pfund.models.model_base import BaseModel
from pfund.utils.utils import short_path


class PytorchModel(BaseModel):
    def load(self) -> dict:
        import torch
        
        file_path = self._get_file_path(extension='.pt')
        if os.path.exists(file_path):
            obj = torch.load(file_path)
            self.ml_model.load_state_dict(obj['state_dict'])
            self._assert_no_missing_datas(obj)
            self.logger.debug(f"loaded trained '{self.name}' from {short_path(file_path)}")
            return obj
        return {}
    
    def dump(self, obj: dict[str, any] | None=None):
        import torch
        
        if obj is None:
            obj = {}
        obj.update({
            "state_dict": self.ml_model.state_dict(),
            'datas': self.datas,
            # TODO, refer to model_base, e.g. dump self.datas
            # 'dataset_periods': {
            #     'train_period': "2020-01-01 to 2020-12-31",
            #     'val_period': ...,
            #     'test_period': ...,
            # },
            # "data_info": {
            #     'data_source': ...,
            #     'resolution': ...,
            # }
        })
        file_path = self._get_file_path(extension='.pt')
        # TODO: need to dump datasets (parquet.gz) as well?
        torch.save(obj, file_path)
        self.logger.debug(f"dumped trained '{self.name}' to {short_path(file_path)}")
        
    def predict(
        self, 
        X: torch.Tensor | pd.DataFrame | pl.LazyFrame, 
        *args, 
        **kwargs
    ) -> torch.Tensor:
        try:
            import pandas as pd
        except ImportError:
            pd = None
        import torch
            
        if pd is not None and isinstance(X, pd.DataFrame):
            X = torch.tensor( X.to_numpy(), dtype=torch.float32 )
        elif pl is not None and isinstance(X, pl.LazyFrame):
            X = torch.tensor( X.collect().to_numpy(), dtype=torch.float32 )
        else:
            raise ValueError(f"Unsupported data type: {type(X)}")
        pred_y = self.ml_model(X, *args, **kwargs)

        if not self._signal_cols:
            num_cols = pred_y.shape[-1]
            signal_cols = self.get_default_signal_cols(num_cols)
            self.set_signal_cols(signal_cols)
        return pred_y
