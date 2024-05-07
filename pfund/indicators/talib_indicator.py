import numpy as np

try:
    import pandas as pd
    import polars as pl
except ImportError:
    pass

from pfund.indicators.indicator_base import TalibFunction, BaseIndicator


class TalibIndicator(BaseIndicator):
    def __init__(self, indicator: TalibFunction, *args, **kwargs):
        '''
        indicator:
            from talib import abstract as talib
            e.g. indicator = talib.SMA
        '''
        if 'min_data_points' not in kwargs:
            if 'timeperiod' in kwargs:
                kwargs['min_data_points'] = kwargs['timeperiod']
            else:
                default_params = indicator.info['parameters']
                if 'timeperiod' in default_params:
                    default_timeperiod = default_params['timeperiod']
                    kwargs['min_data_points'] = default_timeperiod
        super().__init__(indicator, *args, **kwargs)
            
    def get_indicator_info(self):
        return self.indicator.info
    
    def get_indicator_name(self):
        return self.get_indicator_info()['name']
    
    def get_indicator_params(self):
        return self.get_indicator_info()['parameters']
    
    def _predict_pandas(self, X: pd.DataFrame) -> np.ndarray:
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
        pass
    
    def on_start(self):
        default_params = {k: v for k, v in self.get_indicator_params().items() if k not in self._kwargs}
        if default_params:
            self.logger.warning(f'talib indicator {self.name} is using default parameters {default_params}')
        super().on_start()
        