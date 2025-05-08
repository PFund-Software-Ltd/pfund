from __future__ import annotations
from typing import TYPE_CHECKING, Any, Callable
if TYPE_CHECKING:
    import pandas as pd
    import polars as pl
    from pfund.indicators.talib_indicator import TalibFunction
    TaFunction = Callable[..., Any]  # Type for functions from the 'ta' library

from pfund.models.model_base import BaseModel


class BaseIndicator(BaseModel):
    def __init__(self, indicator: TalibFunction, *args, **kwargs):
        from pfeed.enums import DataTool
        
        super().__init__(indicator, *args, **kwargs)
        
        if self.data_tool.name == DataTool.pandas:
            self.predict = self._predict_pandas
        elif self.data_tool.name == DataTool.polars:
            self.predict = self._predict_polars
        else:
            raise ValueError(f'Unsupported data tool: {self.data_tool.name.value}')
    
    def get_default_name(self):
        return self.get_indicator_name()
    
    def predict(self, X: pd.DataFrame | pl.LazyFrame, *args, **kwargs):
        raise NotImplementedError
    indicate = predict

    def featurize(self) -> pd.DataFrame | pl.LazyFrame:
        return self.get_df()
    
    @property
    def indicator(self):
        return self.model

    def load(self) -> dict:
        # since model is None when dumping, 
        # use the initialized model to avoid model=None after loading
        indicator = self.model
        obj: dict = super().load()  # -> self.model = None after loading
        self.model = indicator
        return obj
    
    def dump(self, obj: dict[str, Any] | None=None):
        # NOTE: model is indicator (function of talib.abstract, e.g. abstract.SMA), 
        # which is not serializable, so make it None before dumping
        self.model = None
        super().dump(obj)
    
    def to_dict(self):
        indicator_dict = super().to_dict()
        indicator_dict['model'] = None
        return indicator_dict