"""Equivalence tests: vectorized mode vs hybrid mode.

Core principle:
  When there are NO sequential dependencies, the hybrid (bar-by-bar numba) loop
  should produce IDENTICAL results to the vectorized (cumsum-based) computation.

Key rule for close conditions (SL/TP/TW):
  Vectorized mode always blocks re-entry after SL/TP/TW in the same signal streak
  (limitation #3). To make hybrid reproduce this, we use first_only=True so the
  hybrid kernel also blocks re-entry via has_traded_in_streak.

Trailing stop is NOT tested here — it's hybrid-only and has its own test file.
"""
import numpy as np
import pandas as pd
import polars as pl
import pytest

from pfund.enums import BacktestMode
from pfund._backtest.pandas import BacktestDataFrame as PandasBTDF
from pfund._backtest.polars import BacktestDataFrame as PolarsBTDF


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pandas_df(n: int, *, close: np.ndarray | None = None, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if close is None:
        close = 100.0 + np.cumsum(rng.normal(0, 0.1, n))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) + rng.uniform(0, 0.5, n)
    low = np.minimum(open_, close) - rng.uniform(0, 0.5, n)
    volume = rng.uniform(100, 1000, n)
    return pd.DataFrame({'open': open_, 'high': high, 'low': low, 'close': close, 'volume': volume})


def _make_polars_df(n: int, *, close: np.ndarray | None = None, seed: int = 42) -> pl.DataFrame:
    pdf = _make_pandas_df(n, close=close, seed=seed)
    return pl.from_pandas(pdf)


def _run_pandas_vectorized(base_df: pd.DataFrame, signal: np.ndarray, **kwargs) -> pd.DataFrame:
    df = PandasBTDF(base_df.copy(), backtest_mode=BacktestMode.VECTORIZED)
    df = df.create_signal(signal=pd.Series(signal, index=df.index))
    open_kw = {k: v for k, v in kwargs.items() if k in ('order_price', 'order_quantity', 'first_only', 'long_only', 'short_only')}
    close_kw = {k: v for k, v in kwargs.items() if k in ('take_profit', 'stop_loss', 'time_window')}
    df = df.open_position(**open_kw)
    df = df.close_position(**close_kw)
    return pd.DataFrame(df)


def _run_pandas_hybrid(base_df: pd.DataFrame, signal: np.ndarray, **kwargs) -> pd.DataFrame:
    df = PandasBTDF(base_df.copy(), backtest_mode=BacktestMode.HYBRID)
    df = df.create_signal(signal=pd.Series(signal, index=df.index))
    open_kw = {k: v for k, v in kwargs.items() if k in ('order_price', 'order_quantity', 'first_only', 'long_only', 'short_only')}
    close_kw = {k: v for k, v in kwargs.items() if k in ('take_profit', 'stop_loss', 'time_window')}
    df = df.open_position(**open_kw)
    df = df.close_position(**close_kw)
    df = df.backtest_loop()
    return pd.DataFrame(df)


def _np_signal_to_polars(signal: np.ndarray) -> pl.Series:
    """Convert numpy signal array to polars Series, mapping NaN → null."""
    values = [None if np.isnan(v) else v for v in signal]
    return pl.Series('signal', values, dtype=pl.Float64)


def _run_polars_vectorized(base_df: pl.DataFrame, signal: np.ndarray, **kwargs) -> pd.DataFrame:
    df = PolarsBTDF(base_df.clone(), backtest_mode=BacktestMode.VECTORIZED)
    sig = _np_signal_to_polars(signal)
    df = df.create_signal(signal=sig)
    open_kw = {k: v for k, v in kwargs.items() if k in ('order_price', 'order_quantity', 'first_only', 'long_only', 'short_only')}
    close_kw = {k: v for k, v in kwargs.items() if k in ('take_profit', 'stop_loss', 'time_window')}
    df = df.open_position(**open_kw)
    df = df.close_position(**close_kw)
    return df.to_pandas()


def _run_polars_hybrid(base_df: pl.DataFrame, signal: np.ndarray, **kwargs) -> pd.DataFrame:
    df = PolarsBTDF(base_df.clone(), backtest_mode=BacktestMode.HYBRID)
    sig = _np_signal_to_polars(signal)
    df = df.create_signal(signal=sig)
    open_kw = {k: v for k, v in kwargs.items() if k in ('order_price', 'order_quantity', 'first_only', 'long_only', 'short_only')}
    close_kw = {k: v for k, v in kwargs.items() if k in ('take_profit', 'stop_loss', 'time_window')}
    df = df.open_position(**open_kw)
    df = df.close_position(**close_kw)
    df = df.backtest_loop()
    return df.to_pandas()


COMPARE_COLS = ['order_price', 'order_size', 'trade_price', 'trade_size', 'position', 'avg_price']


def _assert_frames_match(vec_df: pd.DataFrame, hyb_df: pd.DataFrame, cols: list[str] | None = None, rtol: float = 1e-9):
    """Assert that two dataframes match on the given columns."""
    if cols is None:
        cols = COMPARE_COLS
    for col in cols:
        if col not in vec_df.columns and col not in hyb_df.columns:
            continue
        v = vec_df[col].values.astype(float) if col in vec_df.columns else np.full(len(vec_df), np.nan)
        h = hyb_df[col].values.astype(float) if col in hyb_df.columns else np.full(len(hyb_df), np.nan)
        both_nan = np.isnan(v) & np.isnan(h)
        mismatch_nan = np.isnan(v) != np.isnan(h)
        if mismatch_nan.any():
            idx = np.where(mismatch_nan)[0]
            raise AssertionError(
                f"Column '{col}': NaN mismatch at rows {idx.tolist()[:10]}\n"
                f"  vectorized: {v[idx[:5]]}\n"
                f"  hybrid:     {h[idx[:5]]}"
            )
        both_finite = ~np.isnan(v) & ~np.isnan(h)
        if both_finite.any():
            if not np.allclose(v[both_finite], h[both_finite], rtol=rtol, equal_nan=True):
                diff_mask = ~np.isclose(v, h, rtol=rtol, equal_nan=True) & both_finite
                idx = np.where(diff_mask)[0]
                raise AssertionError(
                    f"Column '{col}': value mismatch at rows {idx.tolist()[:10]}\n"
                    f"  vectorized: {v[idx[:5]]}\n"
                    f"  hybrid:     {h[idx[:5]]}"
                )


def _assert_stop_price_match(vec_df: pd.DataFrame, hyb_df: pd.DataFrame, rtol: float = 1e-9):
    """Assert stop_price columns match (both may or may not exist)."""
    vec_has = 'stop_price' in vec_df.columns
    hyb_has = 'stop_price' in hyb_df.columns
    if not vec_has and not hyb_has:
        return
    v = vec_df['stop_price'].values.astype(float) if vec_has else np.full(len(vec_df), np.nan)
    h = hyb_df['stop_price'].values.astype(float) if hyb_has else np.full(len(hyb_df), np.nan)
    both_nan = np.isnan(v) & np.isnan(h)
    both_finite = ~np.isnan(v) & ~np.isnan(h)
    mismatch_nan = np.isnan(v) != np.isnan(h)
    if mismatch_nan.any():
        idx = np.where(mismatch_nan)[0]
        raise AssertionError(
            f"Column 'stop_price': NaN mismatch at rows {idx.tolist()[:10]}\n"
            f"  vectorized: {v[idx[:5]]}\n"
            f"  hybrid:     {h[idx[:5]]}"
        )
    if both_finite.any():
        if not np.allclose(v[both_finite], h[both_finite], rtol=rtol, equal_nan=True):
            diff_mask = ~np.isclose(v, h, rtol=rtol, equal_nan=True) & both_finite
            idx = np.where(diff_mask)[0]
            raise AssertionError(
                f"Column 'stop_price': value mismatch at rows {idx.tolist()[:10]}\n"
                f"  vectorized: {v[idx[:5]]}\n"
                f"  hybrid:     {h[idx[:5]]}"
            )


# ===========================================================================
# NO CLOSE CONDITIONS — both modes should match without first_only
# ===========================================================================

class TestNoCloseConditions:
    """Without SL/TP/TW there's no path dependency, so both modes match."""

    @pytest.mark.parametrize("backend", ["pandas", "polars"])
    def test_alternating_signals(self, backend):
        n = 20
        signal = np.array([1, -1] * (n // 2), dtype=float)
        if backend == "pandas":
            base = _make_pandas_df(n)
            vec = _run_pandas_vectorized(base, signal)
            hyb = _run_pandas_hybrid(base, signal)
        else:
            base = _make_polars_df(n)
            vec = _run_polars_vectorized(base, signal)
            hyb = _run_polars_hybrid(base, signal)
        _assert_frames_match(vec, hyb)

    @pytest.mark.parametrize("backend", ["pandas", "polars"])
    def test_sparse_signals_with_gaps(self, backend):
        n = 30
        signal = np.full(n, np.nan)
        signal[3] = 1.0
        signal[10] = -1.0
        signal[18] = 1.0
        signal[25] = -1.0
        if backend == "pandas":
            base = _make_pandas_df(n)
            vec = _run_pandas_vectorized(base, signal)
            hyb = _run_pandas_hybrid(base, signal)
        else:
            base = _make_polars_df(n)
            vec = _run_polars_vectorized(base, signal)
            hyb = _run_polars_hybrid(base, signal)
        _assert_frames_match(vec, hyb)

    @pytest.mark.parametrize("backend", ["pandas", "polars"])
    def test_first_only_with_streaks(self, backend):
        """Streaks of same signal with first_only — only first trade per streak."""
        n = 20
        signal = np.array([1]*5 + [-1]*5 + [1]*5 + [-1]*5, dtype=float)
        if backend == "pandas":
            base = _make_pandas_df(n)
            vec = _run_pandas_vectorized(base, signal, first_only=True)
            hyb = _run_pandas_hybrid(base, signal, first_only=True)
        else:
            base = _make_polars_df(n)
            vec = _run_polars_vectorized(base, signal, first_only=True)
            hyb = _run_polars_hybrid(base, signal, first_only=True)
        _assert_frames_match(vec, hyb)

    @pytest.mark.parametrize("backend", ["pandas", "polars"])
    def test_long_only_alternating(self, backend):
        n = 20
        signal = np.array([1, -1] * (n // 2), dtype=float)
        if backend == "pandas":
            base = _make_pandas_df(n)
            vec = _run_pandas_vectorized(base, signal, long_only=True)
            hyb = _run_pandas_hybrid(base, signal, long_only=True)
        else:
            base = _make_polars_df(n)
            vec = _run_polars_vectorized(base, signal, long_only=True)
            hyb = _run_polars_hybrid(base, signal, long_only=True)
        _assert_frames_match(vec, hyb)

    @pytest.mark.parametrize("backend", ["pandas", "polars"])
    def test_short_only_alternating(self, backend):
        n = 20
        signal = np.array([-1, 1] * (n // 2), dtype=float)
        if backend == "pandas":
            base = _make_pandas_df(n)
            vec = _run_pandas_vectorized(base, signal, short_only=True)
            hyb = _run_pandas_hybrid(base, signal, short_only=True)
        else:
            base = _make_polars_df(n)
            vec = _run_polars_vectorized(base, signal, short_only=True)
            hyb = _run_polars_hybrid(base, signal, short_only=True)
        _assert_frames_match(vec, hyb)

    @pytest.mark.parametrize("backend", ["pandas", "polars"])
    def test_quantity_greater_than_one(self, backend):
        n = 20
        signal = np.array([1, -1] * (n // 2), dtype=float)
        if backend == "pandas":
            base = _make_pandas_df(n)
            vec = _run_pandas_vectorized(base, signal, order_quantity=3)
            hyb = _run_pandas_hybrid(base, signal, order_quantity=3)
        else:
            base = _make_polars_df(n)
            vec = _run_polars_vectorized(base, signal, order_quantity=3)
            hyb = _run_polars_hybrid(base, signal, order_quantity=3)
        _assert_frames_match(vec, hyb)

    @pytest.mark.parametrize("backend", ["pandas", "polars"])
    def test_streaks_no_first_only_no_close(self, backend):
        """Repeated same-direction signals grow position. No close conditions → no path dependency."""
        n = 20
        signal = np.array([1]*5 + [-1]*5 + [1]*5 + [-1]*5, dtype=float)
        if backend == "pandas":
            base = _make_pandas_df(n)
            vec = _run_pandas_vectorized(base, signal)
            hyb = _run_pandas_hybrid(base, signal)
        else:
            base = _make_polars_df(n)
            vec = _run_polars_vectorized(base, signal)
            hyb = _run_polars_hybrid(base, signal)
        _assert_frames_match(vec, hyb)


# ===========================================================================
# WITH CLOSE CONDITIONS + first_only=True — both modes should match
# Vectorized blocks re-entry after SL/TP/TW (limitation #3).
# first_only=True makes hybrid also block re-entry via has_traded_in_streak.
# ===========================================================================

class TestWithCloseConditionsFirstOnly:
    """With close conditions and first_only=True, both modes should be equivalent."""

    @pytest.mark.parametrize("backend", ["pandas", "polars"])
    def test_stop_loss_alternating_streaks(self, backend):
        """SL triggers during buy streak, signal changes to sell, SL triggers again.
        first_only=True ensures one trade per streak in both modes."""
        n = 30
        close = np.array([
            100, 101, 102, 103, 104,      # 0-4:  up (buy streak)
            100, 98, 96, 94, 92,           # 5-9:  sharp drop triggers SL
            90, 89, 88, 87, 86,            # 10-14: continued down (sell streak)
            90, 92, 94, 96, 98,            # 15-19: recovery triggers SL on short
            100, 101, 102, 103, 104,       # 20-24: up (buy streak)
            100, 98, 96, 94, 92,           # 25-29: drop triggers SL
        ], dtype=float)
        signal = np.array([1]*10 + [-1]*10 + [1]*10, dtype=float)
        kwargs = dict(first_only=True, stop_loss=0.03)
        if backend == "pandas":
            base = _make_pandas_df(n, close=close)
            vec = _run_pandas_vectorized(base, signal, **kwargs)
            hyb = _run_pandas_hybrid(base, signal, **kwargs)
        else:
            base = _make_polars_df(n, close=close)
            vec = _run_polars_vectorized(base, signal, **kwargs)
            hyb = _run_polars_hybrid(base, signal, **kwargs)
        _assert_frames_match(vec, hyb)
        _assert_stop_price_match(vec, hyb)

    @pytest.mark.parametrize("backend", ["pandas", "polars"])
    def test_take_profit_alternating_streaks(self, backend):
        """TP triggers during uptrend buy streak, then signal changes."""
        n = 20
        close = np.array([
            100, 101, 102, 103, 104,       # 0-4:  up → TP triggers
            105, 106, 107, 108, 109,        # 5-9:  continued up
            108, 107, 106, 105, 104,        # 10-14: down (sell streak)
            100, 98, 96, 94, 92,            # 15-19: sharp drop → TP on short
        ], dtype=float)
        signal = np.array([1]*10 + [-1]*10, dtype=float)
        kwargs = dict(first_only=True, take_profit=0.02)
        if backend == "pandas":
            base = _make_pandas_df(n, close=close)
            vec = _run_pandas_vectorized(base, signal, **kwargs)
            hyb = _run_pandas_hybrid(base, signal, **kwargs)
        else:
            base = _make_polars_df(n, close=close)
            vec = _run_polars_vectorized(base, signal, **kwargs)
            hyb = _run_polars_hybrid(base, signal, **kwargs)
        _assert_frames_match(vec, hyb)
        _assert_stop_price_match(vec, hyb)

    @pytest.mark.parametrize("backend", ["pandas", "polars"])
    def test_time_window_with_streaks(self, backend):
        """time_window=3 closes position after 3 bars of holding, then blocked in same streak.
        NOTE: stop_price not compared — vectorized reuses it for TW cleanup, hybrid doesn't."""
        n = 20
        close = 100.0 + np.arange(n, dtype=float) * 0.1
        signal = np.array([1]*10 + [-1]*10, dtype=float)
        kwargs = dict(first_only=True, time_window=3)
        if backend == "pandas":
            base = _make_pandas_df(n, close=close)
            vec = _run_pandas_vectorized(base, signal, **kwargs)
            hyb = _run_pandas_hybrid(base, signal, **kwargs)
        else:
            base = _make_polars_df(n, close=close)
            vec = _run_polars_vectorized(base, signal, **kwargs)
            hyb = _run_polars_hybrid(base, signal, **kwargs)
        _assert_frames_match(vec, hyb)

    @pytest.mark.parametrize("backend", ["pandas", "polars"])
    def test_stop_loss_and_take_profit_combined(self, backend):
        """Both SL and TP set. SL has priority when both could trigger."""
        n = 30
        close = np.array([
            100, 101, 102, 103, 104,       # 0-4: up
            100, 97, 95, 93, 91,           # 5-9: drop → SL triggers
            90, 89, 88, 87, 86,            # 10-14: sell streak
            85, 84, 83, 82, 81,            # 15-19: down → TP on short
            80, 82, 84, 86, 88,            # 20-24: recovery
            90, 92, 94, 96, 98,            # 25-29: up → SL on short
        ], dtype=float)
        signal = np.array([1]*10 + [-1]*20, dtype=float)
        kwargs = dict(first_only=True, stop_loss=0.05, take_profit=0.03)
        if backend == "pandas":
            base = _make_pandas_df(n, close=close)
            vec = _run_pandas_vectorized(base, signal, **kwargs)
            hyb = _run_pandas_hybrid(base, signal, **kwargs)
        else:
            base = _make_polars_df(n, close=close)
            vec = _run_polars_vectorized(base, signal, **kwargs)
            hyb = _run_polars_hybrid(base, signal, **kwargs)
        _assert_frames_match(vec, hyb)
        _assert_stop_price_match(vec, hyb)

    @pytest.mark.parametrize("backend", ["pandas", "polars"])
    def test_stop_loss_with_quantity(self, backend):
        """SL with order_quantity > 1 and first_only."""
        n = 20
        close = np.array([
            100, 101, 102, 103, 104,
            100, 97, 95, 93, 91,
            90, 92, 94, 96, 98,
            100, 98, 96, 94, 92,
        ], dtype=float)
        signal = np.array([1]*10 + [-1]*10, dtype=float)
        kwargs = dict(first_only=True, stop_loss=0.05, order_quantity=2)
        if backend == "pandas":
            base = _make_pandas_df(n, close=close)
            vec = _run_pandas_vectorized(base, signal, **kwargs)
            hyb = _run_pandas_hybrid(base, signal, **kwargs)
        else:
            base = _make_polars_df(n, close=close)
            vec = _run_polars_vectorized(base, signal, **kwargs)
            hyb = _run_polars_hybrid(base, signal, **kwargs)
        _assert_frames_match(vec, hyb)

    @pytest.mark.parametrize("backend", ["pandas", "polars"])
    def test_long_only_with_stop_loss(self, backend):
        """long_only + first_only + SL. Signal=1 opens long, signal=-1 closes it."""
        n = 20
        close = np.array([
            100, 101, 102, 103, 104,
            100, 97, 95, 93, 91,
            90, 92, 94, 96, 98,
            100, 101, 102, 103, 104,
        ], dtype=float)
        signal = np.array([1]*5 + [-1]*5 + [1]*5 + [-1]*5, dtype=float)
        kwargs = dict(first_only=True, long_only=True, stop_loss=0.05)
        if backend == "pandas":
            base = _make_pandas_df(n, close=close)
            vec = _run_pandas_vectorized(base, signal, **kwargs)
            hyb = _run_pandas_hybrid(base, signal, **kwargs)
        else:
            base = _make_polars_df(n, close=close)
            vec = _run_polars_vectorized(base, signal, **kwargs)
            hyb = _run_polars_hybrid(base, signal, **kwargs)
        _assert_frames_match(vec, hyb)

    @pytest.mark.parametrize("backend", ["pandas", "polars"])
    def test_short_only_with_stop_loss(self, backend):
        """short_only + first_only + SL."""
        n = 20
        close = np.array([
            100, 99, 98, 97, 96,
            100, 103, 105, 107, 109,
            110, 108, 106, 104, 102,
            100, 99, 98, 97, 96,
        ], dtype=float)
        signal = np.array([-1]*5 + [1]*5 + [-1]*5 + [1]*5, dtype=float)
        kwargs = dict(first_only=True, short_only=True, stop_loss=0.05)
        if backend == "pandas":
            base = _make_pandas_df(n, close=close)
            vec = _run_pandas_vectorized(base, signal, **kwargs)
            hyb = _run_pandas_hybrid(base, signal, **kwargs)
        else:
            base = _make_polars_df(n, close=close)
            vec = _run_polars_vectorized(base, signal, **kwargs)
            hyb = _run_polars_hybrid(base, signal, **kwargs)
        _assert_frames_match(vec, hyb)

    @pytest.mark.parametrize("backend", ["pandas", "polars"])
    def test_time_window_short_streaks(self, backend):
        """Short alternating streaks where TW doesn't trigger because streak is too short."""
        n = 20
        signal = np.array([1]*3 + [-1]*3 + [1]*3 + [-1]*3 + [1]*4 + [-1]*4, dtype=float)
        close = 100.0 + np.arange(n, dtype=float) * 0.05
        kwargs = dict(first_only=True, time_window=5)
        if backend == "pandas":
            base = _make_pandas_df(n, close=close)
            vec = _run_pandas_vectorized(base, signal, **kwargs)
            hyb = _run_pandas_hybrid(base, signal, **kwargs)
        else:
            base = _make_polars_df(n, close=close)
            vec = _run_polars_vectorized(base, signal, **kwargs)
            hyb = _run_polars_hybrid(base, signal, **kwargs)
        _assert_frames_match(vec, hyb)

    @pytest.mark.parametrize("backend", ["pandas", "polars"])
    def test_sl_tp_tw_all_combined(self, backend):
        """All close conditions combined with first_only.
        NOTE: stop_price not compared — vectorized reuses it for TW cleanup, hybrid doesn't."""
        n = 40
        close = np.concatenate([
            100 + np.arange(10) * 0.5,     # gentle up
            105 - np.arange(10) * 1.5,      # sharp down
            90 + np.arange(10) * 0.3,        # gentle up
            93 - np.arange(10) * 0.8,        # moderate down
        ])
        signal = np.array([1]*10 + [-1]*10 + [1]*10 + [-1]*10, dtype=float)
        kwargs = dict(first_only=True, stop_loss=0.05, take_profit=0.03, time_window=5)
        if backend == "pandas":
            base = _make_pandas_df(n, close=close)
            vec = _run_pandas_vectorized(base, signal, **kwargs)
            hyb = _run_pandas_hybrid(base, signal, **kwargs)
        else:
            base = _make_polars_df(n, close=close)
            vec = _run_polars_vectorized(base, signal, **kwargs)
            hyb = _run_polars_hybrid(base, signal, **kwargs)
        _assert_frames_match(vec, hyb)


# ===========================================================================
# ALTERNATING SIGNALS WITH first_only — every bar is a new streak
# ===========================================================================

class TestFirstOnlyAlternating:
    """With alternating [1,-1,1,-1,...] and first_only=True, every bar is a new streak.
    Both modes should produce orders at every signal bar."""

    @pytest.mark.parametrize("backend", ["pandas", "polars"])
    def test_basic_alternating_first_only(self, backend):
        n = 8
        signal = np.array([1, -1, 1, -1, 1, -1, 1, -1], dtype=float)
        kwargs = dict(first_only=True)
        if backend == "pandas":
            base = _make_pandas_df(n)
            vec = _run_pandas_vectorized(base, signal, **kwargs)
            hyb = _run_pandas_hybrid(base, signal, **kwargs)
        else:
            base = _make_polars_df(n)
            vec = _run_polars_vectorized(base, signal, **kwargs)
            hyb = _run_polars_hybrid(base, signal, **kwargs)
        _assert_frames_match(vec, hyb)

    @pytest.mark.parametrize("backend", ["pandas", "polars"])
    def test_alternating_first_only_with_sl(self, backend):
        """Alternating signals with first_only and SL. Since signal changes every bar,
        SL never triggers (position held for only 1 bar with tiny price moves)."""
        n = 20
        signal = np.array([1, -1] * (n // 2), dtype=float)
        kwargs = dict(first_only=True, stop_loss=0.01)
        if backend == "pandas":
            base = _make_pandas_df(n)
            vec = _run_pandas_vectorized(base, signal, **kwargs)
            hyb = _run_pandas_hybrid(base, signal, **kwargs)
        else:
            base = _make_polars_df(n)
            vec = _run_polars_vectorized(base, signal, **kwargs)
            hyb = _run_polars_hybrid(base, signal, **kwargs)
        _assert_frames_match(vec, hyb)


# ===========================================================================
# HYBRID KERNEL SANITY — basic behavioral checks
# ===========================================================================

class TestHybridKernelSanity:
    """Direct behavioral checks on the hybrid kernel."""

    def test_no_trade_on_first_bar(self):
        n = 10
        base = _make_pandas_df(n)
        signal = np.ones(n)
        hyb = _run_pandas_hybrid(base, signal)
        assert np.isnan(hyb['trade_price'].iloc[0])
        assert np.isnan(hyb['trade_size'].iloc[0])

    def test_order_placed_on_signal_bar(self):
        n = 10
        base = _make_pandas_df(n)
        signal = np.full(n, np.nan)
        signal[3] = 1.0
        hyb = _run_pandas_hybrid(base, signal)
        assert not np.isnan(hyb['order_price'].iloc[3])
        assert not np.isnan(hyb['order_size'].iloc[3])
        assert hyb['order_size'].iloc[3] > 0

    def test_market_order_fills_at_next_bar_close(self):
        """Default fill_price='close' means market fills at prev_close (= order bar's close)."""
        n = 10
        base = _make_pandas_df(n)
        signal = np.full(n, np.nan)
        signal[2] = 1.0
        hyb = _run_pandas_hybrid(base, signal)
        if not np.isnan(hyb['trade_price'].iloc[3]):
            assert hyb['trade_price'].iloc[3] == base['close'].iloc[2]

    def test_position_tracks_correctly(self):
        n = 10
        base = _make_pandas_df(n)
        signal = np.full(n, np.nan)
        signal[1] = 1.0
        signal[5] = -1.0
        hyb = _run_pandas_hybrid(base, signal)
        pos = hyb['position'].values.astype(float)
        assert pos[0] == 0
        assert pos[1] == 0
        if not np.isnan(hyb['trade_price'].iloc[2]):
            assert pos[2] == 1.0

    def test_precondition_errors(self):
        """backtest_loop() without open_position or close_position should raise ValueError."""
        n = 5
        base = _make_pandas_df(n)
        signal = np.array([1, -1, 1, -1, 1], dtype=float)

        df = PandasBTDF(base.copy(), backtest_mode=BacktestMode.HYBRID)
        df = df.create_signal(signal=pd.Series(signal))
        df = df.close_position()
        with pytest.raises(ValueError, match="open_position"):
            df.backtest_loop()

        df2 = PandasBTDF(base.copy(), backtest_mode=BacktestMode.HYBRID)
        df2 = df2.create_signal(signal=pd.Series(signal))
        df2 = df2.open_position()
        with pytest.raises(ValueError, match="close_position"):
            df2.backtest_loop()


# ===========================================================================
# EDGE CASES
# ===========================================================================

class TestEdgeCases:
    """Boundary conditions that might trip up one mode but not the other."""

    @pytest.mark.parametrize("backend", ["pandas", "polars"])
    def test_all_nan_signal(self, backend):
        """All-NaN signal → no trades in either mode."""
        n = 10
        signal = np.full(n, np.nan)
        if backend == "pandas":
            base = _make_pandas_df(n)
            vec = _run_pandas_vectorized(base, signal)
            hyb = _run_pandas_hybrid(base, signal)
        else:
            base = _make_polars_df(n)
            vec = _run_polars_vectorized(base, signal)
            hyb = _run_polars_hybrid(base, signal)
        _assert_frames_match(vec, hyb)
        # Verify no trades happened
        assert np.all(np.isnan(hyb['trade_price'].values.astype(float)))

    @pytest.mark.parametrize("backend", ["pandas", "polars"])
    def test_single_buy_signal_at_start(self, backend):
        """Signal only on first bar — order placed but never flipped."""
        n = 10
        signal = np.full(n, np.nan)
        signal[0] = 1.0
        if backend == "pandas":
            base = _make_pandas_df(n)
            vec = _run_pandas_vectorized(base, signal)
            hyb = _run_pandas_hybrid(base, signal)
        else:
            base = _make_polars_df(n)
            vec = _run_polars_vectorized(base, signal)
            hyb = _run_polars_hybrid(base, signal)
        _assert_frames_match(vec, hyb)

    @pytest.mark.parametrize("backend", ["pandas", "polars"])
    def test_signal_on_last_bar(self, backend):
        """Signal on the last bar — order placed but no bar to fill it."""
        n = 10
        signal = np.full(n, np.nan)
        signal[n - 1] = 1.0
        if backend == "pandas":
            base = _make_pandas_df(n)
            vec = _run_pandas_vectorized(base, signal)
            hyb = _run_pandas_hybrid(base, signal)
        else:
            base = _make_polars_df(n)
            vec = _run_polars_vectorized(base, signal)
            hyb = _run_polars_hybrid(base, signal)
        _assert_frames_match(vec, hyb)
        # Last bar order should exist but no trade
        pos = hyb['position'].values.astype(float)
        assert pos[-1] == 0.0 or np.isnan(pos[-1])

    @pytest.mark.parametrize("backend", ["pandas", "polars"])
    def test_two_bars_minimal(self, backend):
        """Minimum viable scenario: 2 bars with a signal on bar 0."""
        n = 2
        signal = np.array([1.0, np.nan])
        if backend == "pandas":
            base = _make_pandas_df(n)
            vec = _run_pandas_vectorized(base, signal)
            hyb = _run_pandas_hybrid(base, signal)
        else:
            base = _make_polars_df(n)
            vec = _run_polars_vectorized(base, signal)
            hyb = _run_polars_hybrid(base, signal)
        _assert_frames_match(vec, hyb)

    @pytest.mark.parametrize("backend", ["pandas", "polars"])
    def test_consecutive_same_signals(self, backend):
        """All bars are buy signals, no first_only. Position should grow every bar."""
        n = 10
        signal = np.ones(n)
        if backend == "pandas":
            base = _make_pandas_df(n)
            vec = _run_pandas_vectorized(base, signal)
            hyb = _run_pandas_hybrid(base, signal)
        else:
            base = _make_polars_df(n)
            vec = _run_polars_vectorized(base, signal)
            hyb = _run_polars_hybrid(base, signal)
        _assert_frames_match(vec, hyb)

    @pytest.mark.parametrize("backend", ["pandas", "polars"])
    def test_long_only_with_sell_signal_only(self, backend):
        """long_only but only sell signals → no positions opened, no trades."""
        n = 10
        signal = -np.ones(n)
        if backend == "pandas":
            base = _make_pandas_df(n)
            vec = _run_pandas_vectorized(base, signal, long_only=True)
            hyb = _run_pandas_hybrid(base, signal, long_only=True)
        else:
            base = _make_polars_df(n)
            vec = _run_polars_vectorized(base, signal, long_only=True)
            hyb = _run_polars_hybrid(base, signal, long_only=True)
        _assert_frames_match(vec, hyb)

    @pytest.mark.parametrize("backend", ["pandas", "polars"])
    def test_short_only_with_buy_signal_only(self, backend):
        """short_only but only buy signals → no positions opened, no trades."""
        n = 10
        signal = np.ones(n)
        if backend == "pandas":
            base = _make_pandas_df(n)
            vec = _run_pandas_vectorized(base, signal, short_only=True)
            hyb = _run_pandas_hybrid(base, signal, short_only=True)
        else:
            base = _make_polars_df(n)
            vec = _run_polars_vectorized(base, signal, short_only=True)
            hyb = _run_polars_hybrid(base, signal, short_only=True)
        _assert_frames_match(vec, hyb)

    @pytest.mark.parametrize("backend", ["pandas", "polars"])
    def test_rapid_flips_with_sl(self, backend):
        """Quick signal flips with SL + first_only. Wide SL ensures it never triggers
        with small price moves, so both modes should agree."""
        n = 20
        signal = np.array([1, -1] * (n // 2), dtype=float)
        kwargs = dict(first_only=True, stop_loss=0.10)
        if backend == "pandas":
            base = _make_pandas_df(n)
            vec = _run_pandas_vectorized(base, signal, **kwargs)
            hyb = _run_pandas_hybrid(base, signal, **kwargs)
        else:
            base = _make_polars_df(n)
            vec = _run_polars_vectorized(base, signal, **kwargs)
            hyb = _run_polars_hybrid(base, signal, **kwargs)
        _assert_frames_match(vec, hyb)

    @pytest.mark.parametrize("backend", ["pandas", "polars"])
    def test_first_only_single_bar_streaks(self, backend):
        """Each streak is 1 bar long with first_only — every bar should still trade."""
        n = 6
        signal = np.array([1, -1, 1, -1, 1, -1], dtype=float)
        kwargs = dict(first_only=True)
        if backend == "pandas":
            base = _make_pandas_df(n)
            vec = _run_pandas_vectorized(base, signal, **kwargs)
            hyb = _run_pandas_hybrid(base, signal, **kwargs)
        else:
            base = _make_polars_df(n)
            vec = _run_polars_vectorized(base, signal, **kwargs)
            hyb = _run_polars_hybrid(base, signal, **kwargs)
        _assert_frames_match(vec, hyb)


# ===========================================================================
# RANDOMIZED FUZZ TESTS — stress both modes with random signals/prices
# ===========================================================================

def _random_signal(n: int, rng: np.random.Generator, nan_prob: float = 0.3) -> np.ndarray:
    """Generate a random signal array: 1, -1, or NaN."""
    choices = np.array([1.0, -1.0, np.nan])
    probs = np.array([(1 - nan_prob) / 2, (1 - nan_prob) / 2, nan_prob])
    return rng.choice(choices, size=n, p=probs)


class TestRandomizedNoCloseConditions:
    """Fuzz test: random signals with no close conditions → modes must match."""

    @pytest.mark.parametrize("seed", range(10))
    @pytest.mark.parametrize("backend", ["pandas", "polars"])
    def test_random_signals_no_close(self, backend, seed):
        rng = np.random.default_rng(seed)
        n = 50
        signal = _random_signal(n, rng)
        if backend == "pandas":
            base = _make_pandas_df(n, seed=seed)
            vec = _run_pandas_vectorized(base, signal)
            hyb = _run_pandas_hybrid(base, signal)
        else:
            base = _make_polars_df(n, seed=seed)
            vec = _run_polars_vectorized(base, signal)
            hyb = _run_polars_hybrid(base, signal)
        _assert_frames_match(vec, hyb)

    @pytest.mark.parametrize("seed", range(10))
    @pytest.mark.parametrize("backend", ["pandas", "polars"])
    def test_random_signals_first_only_no_close(self, backend, seed):
        rng = np.random.default_rng(seed)
        n = 50
        signal = _random_signal(n, rng, nan_prob=0.1)
        if backend == "pandas":
            base = _make_pandas_df(n, seed=seed)
            vec = _run_pandas_vectorized(base, signal, first_only=True)
            hyb = _run_pandas_hybrid(base, signal, first_only=True)
        else:
            base = _make_polars_df(n, seed=seed)
            vec = _run_polars_vectorized(base, signal, first_only=True)
            hyb = _run_polars_hybrid(base, signal, first_only=True)
        _assert_frames_match(vec, hyb)


class TestRandomizedWithCloseConditions:
    """Fuzz test: random signals with close conditions + first_only → modes must match."""

    @pytest.mark.parametrize("seed", range(10))
    @pytest.mark.parametrize("backend", ["pandas", "polars"])
    def test_random_sl_first_only(self, backend, seed):
        rng = np.random.default_rng(seed)
        n = 60
        signal = _random_signal(n, rng, nan_prob=0.2)
        kwargs = dict(first_only=True, stop_loss=0.03)
        if backend == "pandas":
            base = _make_pandas_df(n, seed=seed)
            vec = _run_pandas_vectorized(base, signal, **kwargs)
            hyb = _run_pandas_hybrid(base, signal, **kwargs)
        else:
            base = _make_polars_df(n, seed=seed)
            vec = _run_polars_vectorized(base, signal, **kwargs)
            hyb = _run_polars_hybrid(base, signal, **kwargs)
        _assert_frames_match(vec, hyb)

    @pytest.mark.parametrize("seed", range(10))
    @pytest.mark.parametrize("backend", ["pandas", "polars"])
    def test_random_tp_first_only(self, backend, seed):
        rng = np.random.default_rng(seed)
        n = 60
        signal = _random_signal(n, rng, nan_prob=0.2)
        kwargs = dict(first_only=True, take_profit=0.02)
        if backend == "pandas":
            base = _make_pandas_df(n, seed=seed)
            vec = _run_pandas_vectorized(base, signal, **kwargs)
            hyb = _run_pandas_hybrid(base, signal, **kwargs)
        else:
            base = _make_polars_df(n, seed=seed)
            vec = _run_polars_vectorized(base, signal, **kwargs)
            hyb = _run_polars_hybrid(base, signal, **kwargs)
        _assert_frames_match(vec, hyb)

    @pytest.mark.parametrize("seed", range(10))
    @pytest.mark.parametrize("backend", ["pandas", "polars"])
    def test_random_sl_tp_combined_first_only(self, backend, seed):
        """SL + TP combined (no TW). TW has known timing divergence with random signals."""
        rng = np.random.default_rng(seed)
        n = 80
        signal = _random_signal(n, rng, nan_prob=0.15)
        kwargs = dict(first_only=True, stop_loss=0.04, take_profit=0.03)
        if backend == "pandas":
            base = _make_pandas_df(n, seed=seed)
            vec = _run_pandas_vectorized(base, signal, **kwargs)
            hyb = _run_pandas_hybrid(base, signal, **kwargs)
        else:
            base = _make_polars_df(n, seed=seed)
            vec = _run_polars_vectorized(base, signal, **kwargs)
            hyb = _run_polars_hybrid(base, signal, **kwargs)
        _assert_frames_match(vec, hyb)


# ===========================================================================
# PANDAS vs POLARS CROSS-VALIDATION — same mode, different backend
# ===========================================================================

class TestPandasPolarsConsistency:
    """Ensure pandas and polars backends produce identical results for the same mode."""

    @pytest.mark.parametrize("seed", range(5))
    def test_vectorized_pandas_vs_polars(self, seed):
        rng = np.random.default_rng(seed)
        n = 40
        signal = _random_signal(n, rng)
        pd_base = _make_pandas_df(n, seed=seed)
        pl_base = _make_polars_df(n, seed=seed)
        pd_vec = _run_pandas_vectorized(pd_base, signal, first_only=True, stop_loss=0.03)
        pl_vec = _run_polars_vectorized(pl_base, signal, first_only=True, stop_loss=0.03)
        _assert_frames_match(pd_vec, pl_vec)

    @pytest.mark.parametrize("seed", range(5))
    def test_hybrid_pandas_vs_polars(self, seed):
        rng = np.random.default_rng(seed)
        n = 40
        signal = _random_signal(n, rng)
        pd_base = _make_pandas_df(n, seed=seed)
        pl_base = _make_polars_df(n, seed=seed)
        pd_hyb = _run_pandas_hybrid(pd_base, signal, first_only=True, stop_loss=0.03)
        pl_hyb = _run_polars_hybrid(pl_base, signal, first_only=True, stop_loss=0.03)
        _assert_frames_match(pd_hyb, pl_hyb)

    @pytest.mark.parametrize("seed", range(5))
    def test_all_four_agree_no_close(self, seed):
        """All 4 combinations (pandas/polars × vectorized/hybrid) should agree with no close conditions."""
        rng = np.random.default_rng(seed)
        n = 40
        signal = _random_signal(n, rng)
        pd_base = _make_pandas_df(n, seed=seed)
        pl_base = _make_polars_df(n, seed=seed)
        pd_vec = _run_pandas_vectorized(pd_base, signal)
        pd_hyb = _run_pandas_hybrid(pd_base, signal)
        pl_vec = _run_polars_vectorized(pl_base, signal)
        pl_hyb = _run_polars_hybrid(pl_base, signal)
        _assert_frames_match(pd_vec, pd_hyb)
        _assert_frames_match(pl_vec, pl_hyb)
        _assert_frames_match(pd_vec, pl_vec)

    @pytest.mark.parametrize("seed", range(5))
    def test_all_four_agree_with_close_first_only(self, seed):
        """All 4 combinations agree with close conditions + first_only."""
        rng = np.random.default_rng(seed)
        n = 60
        signal = _random_signal(n, rng, nan_prob=0.2)
        kwargs = dict(first_only=True, stop_loss=0.04, take_profit=0.03)
        pd_base = _make_pandas_df(n, seed=seed)
        pl_base = _make_polars_df(n, seed=seed)
        pd_vec = _run_pandas_vectorized(pd_base, signal, **kwargs)
        pd_hyb = _run_pandas_hybrid(pd_base, signal, **kwargs)
        pl_vec = _run_polars_vectorized(pl_base, signal, **kwargs)
        pl_hyb = _run_polars_hybrid(pl_base, signal, **kwargs)
        _assert_frames_match(pd_vec, pd_hyb)
        _assert_frames_match(pl_vec, pl_hyb)
        _assert_frames_match(pd_vec, pl_vec)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=long'])
