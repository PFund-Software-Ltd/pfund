# pyright: reportArgumentType=false, reportUnnecessaryComparison=false, reportOperatorIssue=false
from typing import Self

import numpy as np
import pandas as pd


class BacktestDataFrame(pd.DataFrame):
    def create_signal(
        self,
        buy_condition: pd.Series | None=None,
        sell_condition: pd.Series | None=None,
        signal: pd.Series | None=None,
        first_only: bool=False,
    ) -> Self:
        '''
        A signal is defined as a sequence of 1s and -1s, where 1 means a buy signal and -1 means a sell signal.
        Args:
            buy_condition: condition to create a buy signal 1
            sell_condition: condition to create a sell signal -1
            signal: provides self-defined signals, buy_condition and sell_condition are ignored if provided
            first_only: only the first signal is remained in each signal sequence
                useful when only the first signal is treated as a true signal
        '''
        if signal is None:
            if buy_condition is None and sell_condition is None:
                raise ValueError("Either buy or sell condition must be provided")
            
            if buy_condition is not None and sell_condition is not None:
                # Validate non-overlapping
                overlaps = buy_condition & sell_condition
                if overlaps.any():
                    raise ValueError(
                        "Overlapping buy and sell condition detected.\n" +
                        "Please make sure that buy and sell conditions are mutually exclusive."
                    )
                conditions = [buy_condition, sell_condition]
                choices = [1, -1]
            else:
                conditions = [buy_condition if buy_condition is not None else sell_condition]
                choices = [1 if buy_condition is not None else -1]

            # Map conditions to signals: buy→1, sell→-1, neither→NaN
            self['signal'] = np.select(conditions, choices, default=np.nan)
        else:
            assert np.isin(signal.unique(), [1, -1, np.nan]).all(), "'signal' must only contain 1, -1, nan"
            self['signal'] = signal

        # REVIEW: treat nan as a signal too? useful when e.g. 1 -> 1 -> nan (could be a sell signal) -> 1
        # if is_nan_signal:
        #     df['_signal_change'] = df['signal'].fillna(0).diff().ne(0)
        # else:
        self['_signal_change'] = self['signal'].ffill().diff().ne(0)
        
        first_non_nan_idx = self['signal'].first_valid_index()
        if first_non_nan_idx is not None:
            # set the first nan sequence to False
            self.loc[:first_non_nan_idx-1, '_signal_change'] = False
        
        # df['_signal_streak'] = df['_signal_change'].cumsum()
        # # signal streak is nan before the first signal occurs
        # df.loc[:first_non_nan_idx-1, '_signal_streak'] = np.nan
        
        if first_only:
            self['signal'] = np.where(self['_signal_change'], self['signal'], np.nan)
        
        return self
    