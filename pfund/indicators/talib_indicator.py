from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import numpy as np
    try:
        import pandas as pd
        import polars as pl
    except ImportError:
        pass
    from pfund.indicators.indicator_base import TalibFunction

from pfund.indicators.indicator_base import BaseIndicator


class TalibIndicator(BaseIndicator):
    def __init__(self, indicator: TalibFunction, *args, **kwargs):
        '''
        indicator:
            from talib import abstract as talib
            e.g. indicator = talib.SMA
        '''
        super().__init__(indicator, *args, **kwargs)
        if min_data := self._derive_min_data():
            self.set_min_data(min_data)
    
    def _derive_min_data(self) -> int | None:
        '''Derives min_data from indicator parameters. If no specified params, use the default ones'''
        if 'timeperiod' in self._kwargs:
            timeperiod = self._kwargs['timeperiod']
        else:
            default_params = self.get_indicator_params()
            timeperiod = default_params.get('timeperiod', None)
        if timeperiod:
            return int(timeperiod)
            
    def get_indicator_info(self):
        return self.indicator.info
    
    def get_indicator_name(self):
        return self.get_indicator_info()['name']
    
    def get_indicator_params(self):
        return self.get_indicator_info()['parameters']
    
    def _predict_pandas(self, X: pd.DataFrame) -> np.ndarray:
        import pandas as pd

        def _indicate(_X: pd.DataFrame) -> pd.DataFrame:
            _df = self.indicator(_X, *self._args, **self._kwargs)
            if type(_df) is pd.Series:
                _df = _df.to_frame(name=self.get_indicator_name())
            return _df
        
        # if group_data is True, it means X will be passed in per group=(product, resolution)
        if self._group_data:
            df = _indicate(X)
        else:
            grouped_df = X.groupby(self.GROUP).apply(_indicate)
            df = grouped_df.droplevel([0, 1])
            df.sort_index(inplace=True)

        if not self._signal_cols:
            self.set_signal_cols(df.columns.to_list())
        
        return df.to_numpy()

    # TODO
    def _predict_polars(self):
        import polars as pl
        pass
    
    def on_start(self):
        default_params = {k: v for k, v in self.get_indicator_params().items() if k not in self._kwargs}
        if default_params:
            self.logger.warning(f'talib indicator {self.name} is using default parameters {default_params}')
        super().on_start()
    