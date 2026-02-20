"""Trailing stop tests — hybrid mode only (not supported in vectorized).

A trailing stop tracks the best price since entry and closes when price
retraces by the trailing percentage from that peak/trough:
  - Long: stop = best_high * (1 - trailing_stop)
  - Short: stop = best_low * (1 + trailing_stop)
"""
import numpy as np
import pandas as pd
import pytest

from pfund.enums import BacktestMode
from pfund._backtest.pandas import BacktestDataFrame


def _make_df(n: int, *, open=None, high=None, low=None, close=None, volume=None) -> pd.DataFrame:
    return pd.DataFrame({
        'open': open or [100.0] * n,
        'high': high or [101.0] * n,
        'low': low or [99.0] * n,
        'close': close or [100.0] * n,
        'volume': volume or [1000.0] * n,
    })


def _run_hybrid(base_df: pd.DataFrame, signal: np.ndarray, trailing_stop: float | None = None, **kwargs) -> pd.DataFrame:
    df = BacktestDataFrame(base_df.copy(), backtest_mode=BacktestMode.HYBRID)
    df = df.create_signal(signal=pd.Series(signal, index=df.index))
    open_kw = {k: v for k, v in kwargs.items() if k in ('order_price', 'order_quantity', 'first_only', 'long_only', 'short_only')}
    close_kw = {k: v for k, v in kwargs.items() if k in ('take_profit', 'stop_loss', 'time_window')}
    df = df.open_position(**open_kw)
    df = df.close_position(**close_kw)
    df = df.backtest_loop(trailing_stop=trailing_stop)
    return pd.DataFrame(df)


# ============================================================
# Trailing stop as stop loss (price never moves favorably)
# ============================================================

class TestTrailingStopAsSL:

    def test_long_stopped_out_on_drop(self):
        """Long entry, price drops → trailing stop triggers like a fixed SL.

        Bar 0: signal=+1, close=100 → order placed
        Bar 1: fills at prev_close=100. h=100,l=100. best=100. pending_sl=98.0
        Bar 2: low=97 breaches 98.0 → stop fills at 98.0
        """
        n = 4
        df = _make_df(n,
            open=[100.0, 100.0, 100.0, 100.0],
            high=[101.0, 100.0, 100.0, 100.0],
            low=[99.0, 100.0, 97.0, 99.0],
            close=[100.0, 100.0, 98.0, 100.0],
        )
        signal = np.array([1.0, np.nan, np.nan, np.nan])
        result = _run_hybrid(df, signal, trailing_stop=0.02)

        assert result['position'].iloc[1] == 1.0
        assert result['position'].iloc[2] == 0.0
        assert not np.isnan(result['stop_price'].iloc[1])
        assert abs(result['stop_price'].iloc[1] - 98.0) < 1e-9

    def test_short_stopped_out_on_rise(self):
        """Short entry, price rises → trailing stop triggers.

        Bar 0: signal=-1, close=100 → order placed
        Bar 1: fills at prev_close=100. best=100. pending_sl=102.0
        Bar 2: high=103 breaches 102.0 → stop fills at 102.0
        """
        n = 4
        df = _make_df(n,
            open=[100.0, 100.0, 100.0, 100.0],
            high=[101.0, 100.0, 103.0, 101.0],
            low=[99.0, 100.0, 99.0, 99.0],
            close=[100.0, 100.0, 101.0, 100.0],
        )
        signal = np.array([-1.0, np.nan, np.nan, np.nan])
        result = _run_hybrid(df, signal, trailing_stop=0.02)

        assert result['position'].iloc[1] == -1.0
        assert result['position'].iloc[2] == 0.0
        assert abs(result['stop_price'].iloc[1] - 102.0) < 1e-9


# ============================================================
# Trailing stop as profit taker (price runs then retraces)
# ============================================================

class TestTrailingStopAsProfit:

    def test_long_profit_locked_in(self):
        """Price rallies then pulls back → trailing stop locks in profit.

        Bar 0: signal=+1
        Bar 1: fills at prev_close=100. best=100. sl=95.0
        Bar 2: h=110 → best=110. sl=104.5
        Bar 3: h=120 → best=120. sl=114.0
        Bar 4: low=113 < 114.0 → stop at 114.0 (profit of 14)
        """
        n = 6
        df = _make_df(n,
            open=[100.0, 100.0, 105.0, 115.0, 118.0, 100.0],
            high=[101.0, 100.0, 110.0, 120.0, 118.0, 101.0],
            low=[99.0, 100.0, 104.0, 114.0, 113.0, 99.0],
            close=[100.0, 100.0, 108.0, 118.0, 115.0, 100.0],
        )
        signal = np.array([1.0, np.nan, np.nan, np.nan, np.nan, np.nan])
        result = _run_hybrid(df, signal, trailing_stop=0.05)

        assert result['position'].iloc[1] == 1.0
        assert result['position'].iloc[2] == 1.0
        assert result['position'].iloc[3] == 1.0
        assert result['position'].iloc[4] == 0.0
        assert abs(result['stop_price'].iloc[3] - 114.0) < 1e-9
        assert abs(result['trade_price'].iloc[4] - 114.0) < 1e-9

    def test_short_profit_locked_in(self):
        """Price drops then bounces → trailing stop locks in profit.

        Bar 0: signal=-1
        Bar 1: fills at prev_close=100. best=100. sl=105.0
        Bar 2: l=90 → best=90. sl=94.5
        Bar 3: l=80 → best=80. sl=84.0
        Bar 4: high=85 > 84.0 → stop at 84.0 (profit of 16)
        """
        n = 6
        df = _make_df(n,
            open=[100.0, 100.0, 95.0, 85.0, 82.0, 100.0],
            high=[101.0, 100.0, 96.0, 86.0, 85.0, 101.0],
            low=[99.0, 100.0, 90.0, 80.0, 81.0, 99.0],
            close=[100.0, 100.0, 92.0, 82.0, 83.0, 100.0],
        )
        signal = np.array([-1.0, np.nan, np.nan, np.nan, np.nan, np.nan])
        result = _run_hybrid(df, signal, trailing_stop=0.05)

        assert result['position'].iloc[1] == -1.0
        assert result['position'].iloc[2] == -1.0
        assert result['position'].iloc[3] == -1.0
        assert result['position'].iloc[4] == 0.0
        assert abs(result['stop_price'].iloc[3] - 84.0) < 1e-9
        assert abs(result['trade_price'].iloc[4] - 84.0) < 1e-9


# ============================================================
# No trigger — price keeps trending
# ============================================================

class TestTrailingNoTrigger:

    def test_steady_uptrend_no_trigger(self):
        """Each bar makes new high with tiny pullbacks within 5% trail."""
        n = 6
        df = _make_df(n,
            open=[100.0, 100.0, 102.0, 104.0, 106.0, 108.0],
            high=[101.0, 102.0, 104.0, 106.0, 108.0, 110.0],
            low=[99.0, 100.0, 101.0, 103.0, 105.0, 107.0],
            close=[100.0, 101.0, 103.0, 105.0, 107.0, 109.0],
        )
        signal = np.array([1.0, np.nan, np.nan, np.nan, np.nan, np.nan])
        result = _run_hybrid(df, signal, trailing_stop=0.05)

        for i in range(1, n):
            assert result['position'].iloc[i] == 1.0, f"Bar {i}: position should stay 1"


# ============================================================
# Coexistence with take profit
# ============================================================

class TestTrailingWithTP:

    def test_tp_hits_before_trailing_stop(self):
        """TP at 5% triggers before trailing stop at 10%.

        Bar 1: fills at 100. sl=90, tp=105.
        Bar 2: high=106 hits TP. SL (90) not triggered since low doesn't breach.
        """
        n = 4
        df = _make_df(n,
            open=[100.0, 100.0, 102.0, 100.0],
            high=[101.0, 100.0, 106.0, 101.0],
            low=[99.0, 100.0, 101.0, 99.0],
            close=[100.0, 100.0, 105.0, 100.0],
        )
        signal = np.array([1.0, np.nan, np.nan, np.nan])
        result = _run_hybrid(df, signal, trailing_stop=0.10, take_profit=0.05)

        assert result['position'].iloc[1] == 1.0
        assert result['position'].iloc[2] == 0.0
        assert abs(result['trade_price'].iloc[2] - 105.0) < 1e-9

    def test_trailing_hits_before_tp(self):
        """Trailing stop at 3% triggers before TP at 50%.

        Bar 1: fills at 100. sl=97, tp=150.
        Bar 2: h=105 → best=105. sl=101.85. tp=150.
        Bar 3: low=101 < 101.85 → trailing stop triggers.
        """
        n = 5
        df = _make_df(n,
            open=[100.0, 100.0, 102.0, 104.0, 100.0],
            high=[101.0, 100.0, 105.0, 104.0, 101.0],
            low=[99.0, 100.0, 101.0, 101.0, 99.0],
            close=[100.0, 100.0, 104.0, 102.0, 100.0],
        )
        signal = np.array([1.0, np.nan, np.nan, np.nan, np.nan])
        result = _run_hybrid(df, signal, trailing_stop=0.03, take_profit=0.50)

        assert result['position'].iloc[1] == 1.0
        assert result['position'].iloc[2] == 1.0
        assert result['position'].iloc[3] == 0.0
        assert abs(result['stop_price'].iloc[2] - 101.85) < 1e-9


# ============================================================
# Trailing replaces fixed SL
# ============================================================

class TestTrailingOverridesSL:

    def test_trailing_overrides_fixed_sl(self):
        """trailing_stop=0.05 overrides stop_loss=0.01.
        Fixed SL would be 99.0. Trailing SL is 95.0.
        Bar 2 low=98 doesn't trigger trailing (98 > 95), but would trigger fixed SL.
        """
        n = 4
        df = _make_df(n,
            open=[100.0, 100.0, 100.0, 100.0],
            high=[101.0, 100.0, 100.0, 101.0],
            low=[99.0, 100.0, 98.0, 99.0],
            close=[100.0, 100.0, 99.0, 100.0],
        )
        signal = np.array([1.0, np.nan, np.nan, np.nan])
        result = _run_hybrid(df, signal, trailing_stop=0.05, stop_loss=0.01)

        assert result['position'].iloc[1] == 1.0
        assert result['position'].iloc[2] == 1.0


# ============================================================
# Immediate stop (prev_close already breaches)
# ============================================================

class TestTrailingImmediate:

    def test_long_immediate_trailing_stop(self):
        """Bar 1 has large high spike but closes low.
        h=110 → best=110. sl=104.5. close=103 < 104.5.
        Bar 2: prev_c=103 <= 104.5 → immediate stop.
        """
        n = 4
        df = _make_df(n,
            open=[100.0, 100.0, 100.0, 100.0],
            high=[101.0, 110.0, 105.0, 101.0],
            low=[99.0, 99.0, 99.0, 99.0],
            close=[100.0, 103.0, 102.0, 100.0],
        )
        signal = np.array([1.0, np.nan, np.nan, np.nan])
        result = _run_hybrid(df, signal, trailing_stop=0.05)

        assert result['position'].iloc[1] == 1.0
        assert result['position'].iloc[2] == 0.0
        assert abs(result['stop_price'].iloc[1] - 104.5) < 1e-9


# ============================================================
# Best price resets on new position
# ============================================================

class TestBestPriceReset:

    def test_best_price_resets_on_new_position(self):
        """Two separate long positions — second should track from its own entry.

        Bar 1: fills at 100, h=110 → best=110, sl=104.5
        Bar 2: prev_c=103 <= 104.5 → immediate stop. Signal changes to -1.
        Bar 3: signal=+1 (new streak)
        Bar 4: fills at 50. best=50. sl=47.5
        Bar 5: low=48 > 47.5 → stays open. Proves best_price reset.
        """
        n = 7
        df = _make_df(n,
            open=[100.0, 100.0, 100.0, 50.0, 50.0, 50.0, 50.0],
            high=[101.0, 110.0, 105.0, 51.0, 50.0, 50.0, 51.0],
            low=[99.0, 99.0, 99.0, 49.0, 50.0, 48.0, 49.0],
            close=[100.0, 103.0, 102.0, 50.0, 50.0, 49.0, 50.0],
        )
        signal = np.array([1.0, np.nan, -1.0, 1.0, np.nan, np.nan, np.nan])
        result = _run_hybrid(df, signal, trailing_stop=0.05, first_only=True)

        assert result['position'].iloc[1] == 1.0, "Bar 1: first long open"
        assert result['position'].iloc[2] == 0.0, "Bar 2: immediate trailing stop"
        assert result['position'].iloc[4] == 1.0, f"Bar 4: second long should open, got {result['position'].iloc[4]}"
        assert result['position'].iloc[5] == 1.0, f"Bar 5: should stay open (48 > 47.5), got {result['position'].iloc[5]}"


# ============================================================
# Gap-through trailing stop
# ============================================================

class TestTrailingGapThrough:

    def test_long_gap_through_trailing_stop(self):
        """Price gaps down through trailing stop → fills at stop price, not open.

        Bar 0: signal=+1
        Bar 1: fills at 100. best=100. sl=95.0
        Bar 2: h=110 → best=110. sl=104.5
        Bar 3: open=103 (gap down below 104.5) → fills at 104.5
        """
        n = 5
        df = _make_df(n,
            open=[100.0, 100.0, 105.0, 103.0, 100.0],
            high=[101.0, 100.0, 110.0, 104.0, 101.0],
            low=[99.0, 100.0, 104.0, 102.0, 99.0],
            close=[100.0, 100.0, 108.0, 103.0, 100.0],
        )
        signal = np.array([1.0, np.nan, np.nan, np.nan, np.nan])
        result = _run_hybrid(df, signal, trailing_stop=0.05)

        assert result['position'].iloc[1] == 1.0
        assert result['position'].iloc[2] == 1.0
        assert result['position'].iloc[3] == 0.0
        # Gap through: open (103) < stop (104.5), but stop fills at stop price
        assert abs(result['stop_price'].iloc[2] - 104.5) < 1e-9

    def test_short_gap_through_trailing_stop(self):
        """Short position, price gaps up through trailing stop.

        Bar 0: signal=-1
        Bar 1: fills at 100. best=100. sl=105.0
        Bar 2: l=90 → best=90. sl=94.5
        Bar 3: open=96 (gap up above 94.5) → fills at 94.5
        """
        n = 5
        df = _make_df(n,
            open=[100.0, 100.0, 95.0, 96.0, 100.0],
            high=[101.0, 100.0, 96.0, 97.0, 101.0],
            low=[99.0, 100.0, 90.0, 95.0, 99.0],
            close=[100.0, 100.0, 92.0, 96.0, 100.0],
        )
        signal = np.array([-1.0, np.nan, np.nan, np.nan, np.nan])
        result = _run_hybrid(df, signal, trailing_stop=0.05)

        assert result['position'].iloc[1] == -1.0
        assert result['position'].iloc[2] == -1.0
        assert result['position'].iloc[3] == 0.0
        assert abs(result['stop_price'].iloc[2] - 94.5) < 1e-9


# ============================================================
# Multiple sequential positions with trailing stop
# ============================================================

class TestTrailingMultiplePositions:

    def test_long_then_short_both_trailing_stopped(self):
        """Two trades: long stopped out, then short stopped out.

        Bar 0: signal=+1
        Bar 1: fills long at 100. best=100. sl=95.0
        Bar 2: low=94 < 95 → stopped at 95. signal=-1 (new streak)
        Bar 3: fills short at close of bar 2. best tracks low.
        Bar 4+: price rises → short trailing stop triggers
        """
        n = 8
        df = _make_df(n,
            open=[100.0, 100.0, 100.0, 96.0, 95.0, 97.0, 100.0, 100.0],
            high=[101.0, 100.0, 100.0, 96.0, 95.0, 100.0, 101.0, 101.0],
            low=[99.0, 100.0, 94.0, 93.0, 93.0, 95.0, 99.0, 99.0],
            close=[100.0, 100.0, 96.0, 94.0, 94.0, 98.0, 100.0, 100.0],
        )
        signal = np.array([1.0, np.nan, -1.0, np.nan, np.nan, np.nan, np.nan, np.nan])
        result = _run_hybrid(df, signal, trailing_stop=0.05, first_only=True)

        # Long fills at bar 1, gets stopped at bar 2
        assert result['position'].iloc[1] == 1.0
        assert result['position'].iloc[2] == 0.0
        # Short should fill at bar 3
        assert result['position'].iloc[3] == -1.0

    def test_three_sequential_positions(self):
        """Three alternating positions, each with trailing stop.

        Tests that state (best_price, position) resets properly between positions.
        """
        n = 12
        df = _make_df(n,
            # Long: entry at 100, high 110, retrace to stop
            open=[100.0, 100.0, 105.0, 115.0, 115.0,
            # Short: entry at ~108, low 95, bounce to stop
                  108.0, 105.0, 100.0, 95.0, 100.0,
            # Long again: entry at ~102
                  102.0, 102.0],
            high=[101.0, 100.0, 110.0, 115.0, 115.0,
                  108.0, 106.0, 101.0, 96.0, 105.0,
                  103.0, 103.0],
            low=[99.0, 100.0, 104.0, 110.0, 108.0,
                 107.0, 100.0, 95.0, 94.0, 99.0,
                 101.0, 101.0],
            close=[100.0, 100.0, 108.0, 113.0, 108.0,
                   107.0, 102.0, 97.0, 95.0, 102.0,
                   102.0, 102.0],
        )
        signal = np.array([1.0, np.nan, np.nan, np.nan, -1.0,
                           np.nan, np.nan, np.nan, 1.0, np.nan,
                           np.nan, np.nan])
        result = _run_hybrid(df, signal, trailing_stop=0.05, first_only=True)

        # First position: long at bar 1
        assert result['position'].iloc[1] == 1.0


# ============================================================
# Trailing stop with time_window interaction
# ============================================================

class TestTrailingWithTimeWindow:

    def test_time_window_closes_before_trailing(self):
        """time_window=3 closes position before trailing stop triggers.

        Bar 1: fills at 100. best=100.
        Bar 2,3,4: position held for 3 bars → TW closes at bar 4.
        Trailing at 10% would be 90 — never triggered.
        """
        n = 6
        df = _make_df(n,
            open=[100.0, 100.0, 101.0, 102.0, 103.0, 100.0],
            high=[101.0, 101.0, 102.0, 103.0, 104.0, 101.0],
            low=[99.0, 99.0, 100.0, 101.0, 102.0, 99.0],
            close=[100.0, 100.0, 101.0, 102.0, 103.0, 100.0],
        )
        signal = np.array([1.0, np.nan, np.nan, np.nan, np.nan, np.nan])
        result = _run_hybrid(df, signal, trailing_stop=0.10, time_window=3)

        assert result['position'].iloc[1] == 1.0
        assert result['position'].iloc[2] == 1.0
        assert result['position'].iloc[3] == 1.0
        assert result['position'].iloc[4] == 0.0  # TW closes

    def test_trailing_stops_before_time_window(self):
        """Trailing stop triggers before time_window expires.

        Bar 1: fills at 100. best=100. sl=97.
        Bar 2: h=110 → best=110. sl=106.7
        Bar 3: low=105 < 106.7 → trailing stop triggers (TW=10 not yet reached).
        """
        n = 5
        df = _make_df(n,
            open=[100.0, 100.0, 105.0, 108.0, 100.0],
            high=[101.0, 100.0, 110.0, 108.0, 101.0],
            low=[99.0, 100.0, 104.0, 105.0, 99.0],
            close=[100.0, 100.0, 108.0, 106.0, 100.0],
        )
        signal = np.array([1.0, np.nan, np.nan, np.nan, np.nan])
        result = _run_hybrid(df, signal, trailing_stop=0.03, time_window=10)

        assert result['position'].iloc[1] == 1.0
        assert result['position'].iloc[2] == 1.0
        assert result['position'].iloc[3] == 0.0  # Trailing stop triggered


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=long'])
