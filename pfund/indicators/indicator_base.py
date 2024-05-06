from typing import Type, Any, Callable

import numpy as np
import pandas as pd

try:
    from talib import abstract as talib
    TalibFunction = Type[talib.Function]  # If talib is available, use its Function type
except ImportError:
    TalibFunction = Any  # Fallback type if talib is not installed
TaFunction = Callable[..., Any]  # Type for functions from the 'ta' library

from pfund.models.model_base import BaseModel


# FIXME: pandas specific
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
        self._signal_cols = []
        
        if self.engine.data_tool == 'pandas':
            self.predict = self._predict_pandas
        elif self.engine.data_tool == 'polars':
            self.predict = self._predict_polars
        else:
            raise ValueError(f'Unsupported data tool: {self.engine.data_tool}')
    
    def predict(self, *args, **kwargs):
        raise NotImplementedError
    
    @property
    def indicator(self):
        return self.ml_model
    
    def set_signal_columns(self, columns: list[str]):
        self._signal_cols = columns
    
    def load(self):
        # since ml_model is None when dumping, 
        # use the initialized ml_model to avoid ml_model=None after loading
        indicator = self.ml_model
        super().load()  # -> self.ml_model = None after loading
        self.ml_model = indicator
    
    def dump(self, signal: pd.DataFrame):
        # NOTE: ml_model is indicator (function of talib.abstract, e.g. abstract.SMA), 
        # which is not serializable, so make it None before dumping
        self.ml_model = None
        super().dump(signal)
    
    def to_signal(self, X: pd.DataFrame, pred_y: np.ndarray) -> pd.DataFrame:
        return super().to_signal(X, pred_y, columns=self._signal_cols)
    
    def flow(self, is_dump=False) -> pd.DataFrame:
        return super().flow(is_dump=is_dump)
