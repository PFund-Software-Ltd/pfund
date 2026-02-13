import polars as pl


# TODO: maybe create a subclass like SafeFrame(pd.DataFrame) to prevent users from peeking into the future?
# e.g. df['close'] = df['close'].shift(-1) should not be allowed
class BacktestDataFrame(pl.DataFrame):
    pass