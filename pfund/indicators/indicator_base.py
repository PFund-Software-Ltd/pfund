from typing import Type, Any, Callable

import numpy as np

try:
    import pandas as pd
    import polars as pl
except ImportError:
    pass

try:
    from talib import abstract as talib
    TalibFunction = Type[talib.Function]  # If talib is available, use its Function type
except ImportError:
    TalibFunction = Any  # Fallback type if talib is not installed
TaFunction = Callable[..., Any]  # Type for functions from the 'ta' library

from pfund.models.model_base import BaseModel


class BaseIndicator(BaseModel):
    def __init__(self, indicator: TaFunction | TalibFunction, *args, **kwargs):
        '''
        TalibFunction:
            from talib import abstract as talib
            e.g. indicator = talib.SMA
        TaFunction:
            import ta
            - type 1 (ta class):
                e.g. indicator = lambda df: ta.volatility.BollingerBands(close=df['close'], ...)
            - type 2 (ta function):
                e.g. indicator = lambda df: ta.volatility.bollinger_mavg(close=df['close'], ...)
        '''
        super().__init__(indicator, *args, **kwargs)
        self.type = 'indicator'
        
        if self.engine.data_tool == 'pandas':
            self.predict = self._predict_pandas
        elif self.engine.data_tool == 'polars':
            self.predict = self._predict_polars
        else:
            raise ValueError(f'Unsupported data tool: {self.engine.data_tool}')
    
    def predict(self, *args, **kwargs):
        raise NotImplementedError
    indicate = predict
    
    @property
    def indicator(self):
        return self.ml_model

    def load(self) -> dict:
        # since ml_model is None when dumping, 
        # use the initialized ml_model to avoid ml_model=None after loading
        indicator = self.ml_model
        obj: dict = super().load()  # -> self.ml_model = None after loading
        self.ml_model = indicator
        return obj
    
    def dump(self):
        # NOTE: ml_model is indicator (function of talib.abstract, e.g. abstract.SMA), 
        # which is not serializable, so make it None before dumping
        self.ml_model = None
        super().dump()
    