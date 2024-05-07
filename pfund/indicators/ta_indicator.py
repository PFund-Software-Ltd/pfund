import inspect
import re

import numpy as np
try:
    import pandas as pd
    import polars as pl
except ImportError:
    pass

from pfund.indicators.indicator_base import TaFunction, BaseIndicator


class TaIndicator(BaseIndicator):
    def __init__(self, indicator: TaFunction, *args, funcs: list[str] | None=None, **kwargs):
        '''
        indicator: 
            import ta
            - type 1 (ta class):
                e.g. indicator = lambda df: ta.volatility.BollingerBands(close=df['close'], ...)
            - type 2 (ta function):
                e.g. indicator = lambda df: ta.volatility.bollinger_mavg(close=df['close'], ...)
        funcs:
            functions supported by lib 'ta'
            e.g. funcs = ['bollinger_mavg', 'bollinger_hband', 'bollinger_lband']
        '''
        if 'min_data_points' not in kwargs:
            # need to set ml_model to make the functions below work before calling super().__init__()
            self.ml_model = indicator  
            window = self._extract_window_from_indicator_params()
            if window:
                kwargs['min_data_points'] = window
        super().__init__(indicator, *args, funcs=funcs, **kwargs)
    
    def _extract_window_from_indicator_params(self) -> int | None:
        if params := self.get_indicator_params():
            match = re.search(r'window=(\d+)', params)
            if match:
                window = match.group(1)
                return int(window)
    
    def get_indicator_info(self):
        return inspect.getsource(self.indicator)
    
    def get_indicator_name(self) -> str | None:
        info = self.get_indicator_info()
        # Use regex to find patterns that look like a function or class call
        # This regex looks for words with dots in between, potentially resembling a callable, e.g., 'module.class.function'
        pattern = re.compile(r'\b\w+(\.\w+)+\(')
        match = pattern.search(info)

        if match:
            callable_found = match.group()
            # Removing the opening parenthesis as it's part of the regex match
            callable_found = callable_found[:-1]
            return callable_found
    
    def get_indicator_params(self) -> str | None:
        info = self.get_indicator_info()
        
        # Use regex to extract the substring within the first pair of parentheses
        # This pattern will match everything inside the first parentheses after the lambda keyword
        pattern = re.compile(r'lambda [^\(]*\((.*?)\)', re.DOTALL)
        match = pattern.search(info)

        if match:
            parameters = match.group(1)  # This is the string within the first parentheses
            return parameters
    
    def _predict_pandas(self, X: pd.DataFrame) -> np.ndarray:
        funcs = self._kwargs.get('funcs', [])
        ta_type = 'class' if funcs else 'function'
        dfs = []
        if len(self.datas) == 1:
            indicator = self.ml_model(X)
            if ta_type == 'class':
                for func in funcs:
                    df = getattr(indicator, func)()
                    dfs.append(df)
            elif ta_type == 'function':
                df = indicator
                dfs.append(df)
            df = pd.concat(dfs, axis=1)
        else:
            if ta_type == 'class':
                for func in funcs:
                    indicate = lambda df: getattr(self.ml_model(df), func)()
                    grouped_df = X.groupby(level=self._GROUP).apply(indicate)
                    if is_correct := self._check_grouped_df_nlevels(grouped_df):
                        df = grouped_df.droplevel([0, 1])
                        dfs.append(df)
                    else:
                        return
            elif ta_type == 'function':
                indicate = lambda df: self.ml_model(df)
                grouped_df = X.groupby(level=self._GROUP).apply(indicate)
                if is_correct := self._check_grouped_df_nlevels(grouped_df):
                    df = grouped_df.droplevel([0, 1])
                    dfs.append(df)
                else:
                    return
            df = pd.concat(dfs, axis=1)
            df.sort_index(level='ts', inplace=True)
        self.set_signal_cols(df.columns.to_list())
        return df.to_numpy()

    # TODO
    def _predict_polars(self):
        pass