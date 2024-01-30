from typing import Type, Any, Callable

import numpy as np
import pandas as pd

try:
    from talib import abstract as talib
    TALibFunction = Type[talib.Function]  # If talib is available, use its Function type
except ImportError:
    TALibFunction = Any  # Fallback type if talib is not installed
TAFunction = Callable[..., Any]  # Type for functions from the 'ta' library

from pfund.models.model_base import BaseModel


class BaseIndicator(BaseModel):
    def __init__(self, indicator: TAFunction | TALibFunction, *args, **kwargs):
        '''
        TALibFunction:
            from talib import abstract as talib
            e.g. indicator = talib.SMA
        TAFunction:
            import ta
            - type 1 (ta class):
                e.g. indicator = lambda df: ta.volatility.BollingerBands(close=df['close'], ...)
            - type 2 (ta function):
                e.g. indicator = lambda df: ta.volatility.bollinger_mavg(close=df['close'], ...)
        '''
        super().__init__(indicator, *args, **kwargs)
        self._signal_columns = []
    
    @property
    def indicator(self):
        return self.ml_model
    
    def set_signal_columns(self, columns: list[str]):
        self._signal_columns = columns
    
    def load(self, path: str=''):
        # since ml_model is None when dumping, 
        # use the initialized ml_model to avoid ml_model=None after loading
        ml_model = self.ml_model
        super().load(path=path)  # -> self.ml_model = None after loading
        self.ml_model = ml_model
    
    def dump(self, signal: pd.DataFrame, path: str=''):
        # NOTE: ml_model is indicator (function of talib.abstract, e.g. abstract.SMA), 
        # which is not serializable, so make it None before dumping
        self.ml_model = None
        super().dump(signal, path=path)
        
    def to_signal(self, X: pd.DataFrame, pred_y: np.ndarray) -> pd.DataFrame:
        return super().to_signal(X, pred_y, columns=self._signal_columns)
    
    def flow(self, is_dump=False, path: str='') -> pd.DataFrame:
        return super().flow(is_dump=is_dump, path=path)

    def _check_grouped_df_nlevels(self, grouped_df: pd.DataFrame) -> bool:
        # e.g. if self._GROUP is ['product', 'resolution'], after groupby() and applying indicate()
        # the index.nlevels will be 5, and hence len(self._GROUP) * 2 + 1 where the 1 is 'ts'
        expected_nlevels = len(self._GROUP) * 2 + 1
        # NOTE: if not enough data, somehow grouped_df will have smaller .index.nlevels
        if grouped_df.index.nlevels < expected_nlevels:
            is_correct = False
        elif grouped_df.index.nlevels > expected_nlevels:
            raise Exception(f'Unexpected {grouped_df.index.nlevels=} > {expected_nlevels=}')
        else:
            is_correct = True
        return is_correct