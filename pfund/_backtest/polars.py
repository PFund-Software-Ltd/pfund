from typing import Self

import polars as pl


# TODO: test on engine="gpu"
# pl.Config.set_engine_affinity(engine="streaming")


# TODO: maybe create a subclass like SafeFrame(pd.DataFrame) to prevent users from peeking into the future?
# e.g. df['close'] = df['close'].shift(-1) should not be allowed
class BacktestDataFrame(pl.DataFrame):
    def create_signal(
        self,
        buy_condition: pl.Series | None=None,
        sell_condition: pl.Series | None=None,
        signal: pl.Series | None=None,
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
                raise ValueError("Either buy or sell must be provided")
            if buy_condition is not None and sell_condition is not None:
                overlaps = buy_condition & sell_condition
                if overlaps.any():
                    raise ValueError(
                        "Overlapping buy and sell condition detected.\n" + 
                        "Please make sure that buy and sell conditions are mutually exclusive."
                    )
                signal_series = (
                    pl.DataFrame({'_buy': buy_condition, '_sell': sell_condition})
                    .select(
                        pl
                        .when(pl.col('_buy')).then(1)
                        .when(pl.col('_sell')).then(-1)
                        .otherwise(None)
                        .alias('signal')
                    )
                    .get_column('signal')
                )
            else:
                cond = buy_condition if buy_condition is not None else sell_condition
                value = 1 if buy_condition is not None else -1
                signal_series = (
                    pl.DataFrame({'_cond': cond})
                    .select(
                        pl
                        .when(pl.col('_cond')).then(value)
                        .otherwise(None)
                        .alias('signal')
                    )
                    .get_column('signal')
                )
        else:
            unique_vals = signal.drop_nulls().unique()
            assert unique_vals.is_in([1, -1]).all(), "'signal' must only contain 1, -1, null"
            signal_series = signal.alias('signal')

        # _signal_change: True where the forward-filled signal differs from its previous value
        signal_ffill = signal_series.forward_fill()
        signal_change = (
            signal_ffill.ne(signal_ffill.shift(1)).fill_null(True)
            & signal_ffill.is_not_null()
        )

        df = self.with_columns([
            signal_series,
            signal_change.alias('_signal_change'),
        ])

        if first_only:
            df = df.with_columns(
                pl.when(pl.col('_signal_change'))
                .then(pl.col('signal'))
                .otherwise(None)
                .alias('signal')
            )

        return self.__class__(df)