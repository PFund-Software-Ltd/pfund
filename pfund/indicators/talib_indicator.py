import numpy as np
import pandas as pd

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
        return self.ml_model.info
    
    def get_indicator_name(self):
        return self.get_indicator_info()['name']
    
    def get_indicator_params(self):
        return self.get_indicator_info()['parameters']
    
    def predict(self, X: pd.DataFrame) -> np.ndarray | None:
        if len(self.datas) == 1:
            df = self.ml_model(X, *self._args, **self._kwargs)
        else:
            indicate = lambda df: self.ml_model(df, *self._args, **self._kwargs)
            grouped_df = X.groupby(level=self._GROUP).apply(indicate)
            if is_correct := self._check_grouped_df_nlevels(grouped_df):
                df = grouped_df.droplevel([0, 1])
                df.sort_index(level='ts', inplace=True)
            else:
                return
        # convert series to dataframe
        if type(df) is pd.Series:
            df = df.to_frame(name=self.get_indicator_name())
        self.set_signal_columns(df.columns.to_list())
        return df.to_numpy()

    def on_start(self):
        default_params = {k: v for k, v in self.get_indicator_params().items() if k not in self._kwargs}
        if default_params:
            self.logger.warning(f'talib indicator {self.name} is using default parameters {default_params}')
        super().on_start()
        