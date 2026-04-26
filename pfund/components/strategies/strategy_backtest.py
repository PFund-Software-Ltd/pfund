from __future__ import annotations
from typing import TYPE_CHECKING, Any, cast
if TYPE_CHECKING:
    from pfund.typing import StrategyT

import narwhals as nw

import pfund as pf
from pfund._backtest.backtest_mixin import BacktestMixin


def BacktestStrategy(Strategy: type[StrategyT], *args: Any, **kwargs: Any) -> StrategyT:
    class _BacktestStrategy(BacktestMixin, Strategy):
        def __getattr__(self, name: str) -> Any:
            if hasattr(super(), name):
                return getattr(super(), name)
            else:
                class_name = Strategy.__name__
                raise AttributeError(f"'{class_name}' object has no attribute '{name}'")
        
        # FIXME: add sub-strategy
        def add_strategy(self, strategy: StrategyT, name: str='') -> StrategyT:
            strategy: StrategyT = BacktestStrategy(type(strategy), *strategy.__pfund_args__, **strategy.__pfund_kwargs__)
            return super().add_strategy(strategy, name=name)
        
        def add_accounts(self):
            super().add_accounts()
            if self.accounts:
                return
            # add account to each trading venue if no accounts are added
            trading_venues = set(product.trading_venue for product in self.products.values())
            for trading_venue in trading_venues:
                self.add_account(trading_venue=trading_venue)
                
        @staticmethod
        def _postprocess_backtest_df(backtest_df: pf.BacktestDataFrame) -> pf.BacktestDataFrame:
            '''Postprocesses backtest DataFrame by merging internal debug columns into a
            single 'remark' column and dropping them.

            Remark abbreviations:
                sc  = signal_change — signal direction flipped (buy <-> sell)
                pc  = position_change — position closed or flipped (new trade streak)
                ft  = first_trade — first trade in a filtered trade streak
                sl  = stop_loss — position closed by stop loss
                tp  = take_profit — position closed by take profit
                tw  = time_window — position closed by time window expiry

            Columns dropped without remark (pure intermediates):
                _agg_costs, _immediate_stop, _gap_through_stop,
                _stop_side, _trade_side, _first_stop_order
            '''
            REMARK_COLS: dict[str, str] = {
                '_signal_change': 'sc',
                '_position_change': 'pc',
                '_first_trade': 'ft',
                '_stop_loss': 'sl',
                '_take_profit': 'tp',
                '_time_window': 'tw',
            }

            df: nw.DataFrame[Any] = nw.from_native(backtest_df)

            # Build remark from existing boolean columns
            remark_parts: list[nw.Expr] = []
            for col, abbr in REMARK_COLS.items():
                if col in df.columns:
                    remark_parts.append(
                        nw
                        .when(nw.col(col))
                        .then(nw.lit(abbr))
                        .otherwise(nw.lit(''))
                    )

            if remark_parts:
                # Concatenate all parts with comma separator, then strip leading/trailing commas
                concat_expr = remark_parts[0]
                for part in remark_parts[1:]:
                    concat_expr = concat_expr + nw.lit(',') + part
                df = df.with_columns(concat_expr.alias('_remark_raw'))
                # Clean up: remove empty segments by replacing multiple commas and stripping edges
                df = df.with_columns(
                    nw.col('_remark_raw')
                    .str.replace_all(r',{2,}', ',')  # collapse consecutive commas from empty parts, e.g. "sc,,,,tp" -> "sc,tp"
                    .str.strip_chars(',')  # remove leading/trailing commas, e.g. ",sl," -> "sl"
                    .alias('remark')
                )
                df = df.drop('_remark_raw')
            else:
                df = df.with_columns(nw.lit('').alias('remark'))

            # Drop all internal columns (prefixed with '_')
            internal_cols = [col for col in df.columns if col.startswith('_')]
            if internal_cols:
                df = df.drop(*internal_cols)

            # wrap it back into the original BacktestDataFrame class for consistency
            BacktestDataFrame = type(backtest_df)
            return BacktestDataFrame(df.to_native(), backtest_mode=backtest_df._backtest_mode)  # pyright: ignore[reportCallIssue]
        
    try:
        return cast("StrategyT", _BacktestStrategy(*args, **kwargs))
    except TypeError as e:
        if '__init__()' in str(e):
            raise TypeError(
                f'if super().__init__() is called in {Strategy.__name__}.__init__() (which is unnecssary), ' +
                'make sure it is called with args and kwargs, i.e. super().__init__(*args, **kwargs)'
            ) from e
        raise