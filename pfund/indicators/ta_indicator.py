from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Any
if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    import polars as pl
    TaFunction = Callable[..., Any]  # Type for functions from the 'ta' library

import inspect
import re

from pfund.indicators.indicator_base import BaseIndicator


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
        super().__init__(indicator, *args, funcs=funcs, **kwargs)
        if min_data := self._derive_min_data():
            self._set_min_data(min_data)
    
    def _derive_min_data(self) -> int | None:
        '''Derives min_data from indicator parameters.
        Since the params are raw strings, we need to extract the window value from them.
        '''
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
        import pandas as pd

        funcs = self._kwargs.get('funcs', [])
        ta_type = 'class' if funcs else 'function'
        dfs = []
        if len(self.datas) == 1:
            indicator = self.model(X)
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
                    indicate = lambda df: getattr(self.model(df), func)()
                    grouped_df = X.groupby(level=self._GROUP).apply(indicate)
                    if is_correct := self._check_grouped_df_nlevels(grouped_df):
                        df = grouped_df.droplevel([0, 1])
                        dfs.append(df)
                    else:
                        return
            elif ta_type == 'function':
                indicate = lambda df: self.model(df)
                grouped_df = X.groupby(level=self._GROUP).apply(indicate)
                if is_correct := self._check_grouped_df_nlevels(grouped_df):
                    df = grouped_df.droplevel([0, 1])
                    dfs.append(df)
                else:
                    return
            df = pd.concat(dfs, axis=1)
            df.sort_index(level='ts', inplace=True)
    
        if not self._signal_cols:
            self._set_signal_cols(df.columns.to_list())
    
        return df.to_numpy()

    # TODO
    def _predict_polars(self, X):
        pass