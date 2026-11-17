from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import pytest

from pfund.backtest.product_backtest_kernel import backtest_loop_kernel

NAN = np.nan

type Bar = tuple[float, float, float, float]  # (open, high, low, close)
type FloatArray = np.ndarray[Any, np.dtype[np.float64]]


class Result(NamedTuple):
    order_price: FloatArray
    order_size: FloatArray
    trade_price: FloatArray
    trade_size: FloatArray
    position: FloatArray
    avg_price: FloatArray
    stop_price: FloatArray


def _arr(value: Any, n: int, dtype: Any) -> np.ndarray[Any, Any]:
    """Broadcast a scalar to a length-n array, or convert a per-bar sequence."""
    if np.ndim(value) == 0:
        return np.full(n, value, dtype=dtype)
    arr = np.asarray(value, dtype=dtype)
    assert len(arr) == n, f"expected {n} values, got {len(arr)}"
    return arr


def flat(*closes: float) -> list[Bar]:
    """Bars with open == high == low == close."""
    return [(c, c, c, c) for c in closes]


def run(
    bars: list[Bar],
    signal: Any,
    *,
    in_universe: Any = True,
    order_price: Any = None,  # None = each bar's close (the mixin's default)
    order_quantity: Any = 1.0,
    first_only: Any = False,
    long_only: Any = False,
    short_only: Any = False,
    take_profit: Any = NAN,
    stop_loss: Any = NAN,
    time_window: Any = -1,
    trailing_stop: Any = NAN,
    market_fill_at_open: Any = False,
    exit_market_fill_at_open: Any = False,
) -> Result:
    n = len(bars)
    ohlc = np.asarray(bars, dtype=np.float64).reshape(n, 4)
    close = np.ascontiguousarray(ohlc[:, 3])
    outs = backtest_loop_kernel(
        np.ascontiguousarray(ohlc[:, 0]),
        np.ascontiguousarray(ohlc[:, 1]),
        np.ascontiguousarray(ohlc[:, 2]),
        close,
        np.ones(n),
        _arr(signal, n, np.float64),
        _arr(in_universe, n, np.bool_),
        close.copy() if order_price is None else _arr(order_price, n, np.float64),
        _arr(order_quantity, n, np.float64),
        _arr(first_only, n, np.bool_),
        _arr(long_only, n, np.bool_),
        _arr(short_only, n, np.bool_),
        _arr(take_profit, n, np.float64),
        _arr(stop_loss, n, np.float64),
        _arr(time_window, n, np.int64),
        _arr(trailing_stop, n, np.float64),
        _arr(market_fill_at_open, n, np.bool_),
        _arr(exit_market_fill_at_open, n, np.bool_),
        n,
    )
    return Result(*outs)


def eq(actual: FloatArray, expected: list[float]) -> None:
    np.testing.assert_allclose(
        actual, np.asarray(expected, dtype=np.float64), equal_nan=True
    )


class TestBasics:
    def test_no_signal_produces_no_orders_or_trades(self):
        r = run(flat(100, 101, 102), [NAN, NAN, NAN])
        for out in (r.order_price, r.order_size, r.trade_price, r.trade_size):
            eq(out, [NAN, NAN, NAN])
        eq(r.position, [0, 0, 0])
        eq(r.avg_price, [NAN, NAN, NAN])
        eq(r.stop_price, [NAN, NAN, NAN])

    def test_single_bar(self):
        r = run(flat(100), [1.0])
        eq(r.order_size, [1])
        eq(r.trade_size, [NAN])
        eq(r.position, [0])

    def test_order_on_last_bar_never_fills(self):
        r = run(flat(100, 100, 100), [NAN, NAN, 1.0])
        eq(r.order_size, [NAN, NAN, 1])
        eq(r.trade_size, [NAN, NAN, NAN])
        eq(r.position, [0, 0, 0])

    def test_outputs_are_float64_of_length_n(self):
        r = run(flat(100, 100, 100, 100), [1.0, NAN, NAN, NAN])
        for out in r:
            assert out.dtype == np.float64
            assert out.shape == (4,)


class TestMarketOrders:
    def test_fills_next_bar_at_prev_close(self):
        r = run(
            [(100, 100, 100, 100), (103, 105, 99, 104), (104, 104, 104, 104)],
            [1.0, NAN, NAN],
        )
        eq(r.order_price, [100, NAN, NAN])
        eq(r.order_size, [1, NAN, NAN])
        eq(r.trade_price, [NAN, 100, NAN])
        eq(r.trade_size, [NAN, 1, NAN])
        eq(r.position, [0, 1, 1])
        eq(r.avg_price, [NAN, 100, 100])

    def test_fills_next_bar_at_open(self):
        r = run(
            [(100, 100, 100, 100), (103, 105, 99, 104)],
            [1.0, NAN],
            market_fill_at_open=True,
        )
        eq(r.trade_price, [NAN, 103])
        eq(r.avg_price, [NAN, 103])

    def test_fill_policy_is_read_at_the_placement_bar(self):
        bars: list[Bar] = [(100, 100, 100, 100), (103, 105, 99, 104)]
        placed_open = run(bars, [1.0, NAN], market_fill_at_open=[True, False])
        placed_close = run(bars, [1.0, NAN], market_fill_at_open=[False, True])
        eq(placed_open.trade_price, [NAN, 103])
        eq(placed_close.trade_price, [NAN, 100])

    def test_short_market_order(self):
        r = run(flat(100, 100, 100), [-1.0, NAN, NAN])
        eq(r.order_size, [-1, NAN, NAN])
        eq(r.trade_size, [NAN, -1, NAN])
        eq(r.position, [0, -1, -1])
        eq(r.avg_price, [NAN, 100, 100])

    def test_aggressive_limit_is_a_market_order(self):
        # buy limit above the market: fills like a market order, not at the limit
        r = run(flat(100, 100), [1.0, NAN], order_price=110.0)
        eq(r.order_price, [110, NAN])
        eq(r.trade_price, [NAN, 100])

    def test_nan_order_price_places_no_order(self):
        r = run(flat(100, 100, 100), [1.0, 1.0, NAN], order_price=NAN)
        eq(r.order_price, [NAN, NAN, NAN])
        eq(r.order_size, [NAN, NAN, NAN])
        eq(r.position, [0, 0, 0])

    def test_nan_order_quantity_places_no_order(self):
        r = run(flat(100, 100, 100), [1.0, 1.0, NAN], order_quantity=NAN)
        eq(r.order_size, [NAN, NAN, NAN])
        eq(r.position, [0, 0, 0])


class TestLimitOrders:
    def test_long_limit_fills_at_limit_price(self):
        r = run(
            [(100, 100, 100, 100), (100, 101, 94, 96)], [1.0, NAN], order_price=95.0
        )
        eq(r.trade_price, [NAN, 95])
        eq(r.trade_size, [NAN, 1])
        eq(r.avg_price, [NAN, 95])

    def test_long_limit_fills_when_low_touches_exactly(self):
        r = run(
            [(100, 100, 100, 100), (100, 101, 95, 96)], [1.0, NAN], order_price=95.0
        )
        eq(r.trade_price, [NAN, 95])

    def test_short_limit_fills_at_limit_price(self):
        r = run(
            [(100, 100, 100, 100), (100, 106, 99, 100)], [-1.0, NAN], order_price=105.0
        )
        eq(r.trade_price, [NAN, 105])
        eq(r.trade_size, [NAN, -1])

    def test_unreached_limit_is_cancelled(self):
        r = run(
            [(100, 100, 100, 100), (100, 101, 96, 97), (97, 98, 90, 91)],
            [1.0, NAN, NAN],
            order_price=95.0,
        )
        # not re-placed at bar 1 (nan signal), so bar 2's dip does not fill it
        eq(r.trade_price, [NAN, NAN, NAN])
        eq(r.position, [0, 0, 0])

    def test_unreached_limit_is_replaced_while_signal_persists(self):
        r = run(
            [
                (100, 100, 100, 100),
                (100, 101, 96, 97),
                (97, 98, 94, 96),
                (96, 96, 96, 96),
            ],
            [1.0, 1.0, NAN, NAN],
            order_price=95.0,
        )
        eq(r.order_price, [95, 95, NAN, NAN])
        eq(r.trade_price, [NAN, NAN, 95, NAN])
        eq(r.position, [0, 0, 1, 1])

    @pytest.mark.parametrize("fill_at_open", [False, True])
    def test_long_gap_through_fills_at_open(self, fill_at_open: bool):
        r = run(
            [(100, 100, 100, 100), (90, 92, 89, 91)],
            [1.0, NAN],
            order_price=95.0,
            market_fill_at_open=fill_at_open,
        )
        eq(r.trade_price, [NAN, 90])

    @pytest.mark.parametrize("fill_at_open", [False, True])
    def test_short_gap_through_fills_at_open(self, fill_at_open: bool):
        r = run(
            [(100, 100, 100, 100), (110, 111, 109, 110)],
            [-1.0, NAN],
            order_price=105.0,
            market_fill_at_open=fill_at_open,
        )
        eq(r.trade_price, [NAN, 110])


class TestPositionBookkeeping:
    def test_scaling_in_averages_the_price(self):
        r = run(flat(100, 110, 120), [1.0, 1.0, NAN])
        eq(r.trade_price, [NAN, 100, 110])
        eq(r.position, [0, 1, 2])
        eq(r.avg_price, [NAN, 100, 105])

    def test_scaling_in_short_keeps_avg_price_positive(self):
        r = run(flat(100, 110, 120), [-1.0, -1.0, NAN], order_quantity=2.0)
        eq(r.position, [0, -2, -4])
        eq(r.avg_price, [NAN, 100, 105])

    def test_opposite_signal_flips_in_one_trade(self):
        r = run(flat(100, 110, 120), [1.0, -1.0, NAN])
        eq(r.order_size, [1, -2, NAN])
        eq(r.trade_size, [NAN, 1, -2])
        eq(r.position, [0, 1, -1])
        # only the remainder opens the new side, at the flip price
        eq(r.avg_price, [NAN, 100, 110])

    def test_flip_from_short_to_long(self):
        r = run(flat(100, 90, 80), [-1.0, 1.0, NAN], order_quantity=[1.0, 3.0, NAN])
        eq(r.order_size, [-1, 4, NAN])
        eq(r.position, [0, -1, 3])
        eq(r.avg_price, [NAN, 100, 90])

    def test_zero_quantity_opposite_signal_closes_to_flat(self):
        r = run(flat(100, 110, 120), [1.0, -1.0, NAN], order_quantity=[1.0, 0.0, NAN])
        eq(r.order_size, [1, -1, NAN])
        eq(r.position, [0, 1, 0])
        eq(r.avg_price, [NAN, 100, NAN])

    def test_quantity_is_read_at_the_placement_bar(self):
        r = run(flat(100, 100, 100), [1.0, 1.0, NAN], order_quantity=[2.0, 5.0, 9.0])
        eq(r.trade_size, [NAN, 2, 5])


class TestFirstOnly:
    def test_without_first_only_every_signal_bar_trades(self):
        r = run(flat(100, 100, 100, 100), [1.0, 1.0, 1.0, NAN])
        eq(r.position, [0, 1, 2, 3])

    def test_one_trade_per_streak(self):
        r = run(flat(100, 100, 100, 100), [1.0, 1.0, 1.0, NAN], first_only=True)
        eq(r.order_size, [1, NAN, NAN, NAN])
        eq(r.position, [0, 1, 1, 1])

    def test_nan_does_not_break_a_streak(self):
        r = run(
            flat(100, 100, 100, 100, 100), [1.0, NAN, 1.0, 1.0, NAN], first_only=True
        )
        eq(r.order_size, [1, NAN, NAN, NAN, NAN])
        eq(r.position, [0, 1, 1, 1, 1])

    def test_signal_change_starts_a_new_streak(self):
        r = run(
            flat(100, 100, 100, 100, 100), [1.0, 1.0, -1.0, -1.0, NAN], first_only=True
        )
        eq(r.order_size, [1, NAN, -2, NAN, NAN])
        eq(r.position, [0, 1, 1, -1, -1])

    def test_old_streak_fill_does_not_block_the_new_streak(self):
        # bar 0's order fills at bar 1, where the signal has already flipped
        r = run(flat(100, 100, 100, 100), [1.0, -1.0, NAN, NAN], first_only=True)
        eq(r.order_size, [1, -2, NAN, NAN])
        eq(r.position, [0, 1, -1, -1])

    def test_unfilled_limit_keeps_being_replaced(self):
        r = run(
            [
                (100, 100, 100, 100),
                (100, 100, 98, 99),
                (99, 99, 94, 96),
                (96, 96, 90, 91),
            ],
            1.0,
            order_price=95.0,
            first_only=True,
        )
        eq(r.order_price, [95, 95, NAN, NAN])
        eq(r.position, [0, 0, 1, 1])

    def test_no_reentry_after_stop_within_the_same_streak(self):
        r = run(
            [
                (100, 100, 100, 100),
                (100, 100, 100, 100),
                (100, 100, 85, 95),
                (95, 95, 95, 95),
            ],
            1.0,
            first_only=True,
            stop_loss=0.1,
        )
        eq(r.trade_price, [NAN, 100, 90, NAN])
        eq(r.position, [0, 1, 0, 0])

    def test_flag_turned_on_mid_streak_blocks_an_already_traded_streak(self):
        r = run(
            flat(100, 100, 100, 100),
            [1.0, 1.0, 1.0, NAN],
            first_only=[False, False, True, True],
        )
        eq(r.order_size, [1, 1, NAN, NAN])
        eq(r.position, [0, 1, 2, 2])


class TestLongShortOnly:
    def test_long_only_sell_signal_closes_the_long(self):
        r = run(
            flat(100, 100, 100, 100),
            [1.0, -1.0, -1.0, NAN],
            long_only=True,
            order_quantity=3.0,
        )
        eq(r.order_size, [3, -3, NAN, NAN])
        eq(r.position, [0, 3, 0, 0])

    def test_long_only_never_opens_a_short(self):
        r = run(flat(100, 100, 100), [-1.0, -1.0, NAN], long_only=True)
        eq(r.order_size, [NAN, NAN, NAN])
        eq(r.position, [0, 0, 0])

    def test_long_only_close_ignores_nan_quantity(self):
        r = run(
            flat(100, 100, 100),
            [1.0, -1.0, NAN],
            long_only=True,
            order_quantity=[1.0, NAN, NAN],
        )
        eq(r.position, [0, 1, 0])

    def test_long_only_close_with_nan_price_holds(self):
        r = run(
            flat(100, 100, 100),
            [1.0, -1.0, NAN],
            long_only=True,
            order_price=[100.0, NAN, NAN],
        )
        eq(r.order_size, [1, NAN, NAN])
        eq(r.position, [0, 1, 1])

    def test_short_only_buy_signal_closes_the_short(self):
        r = run(
            flat(100, 100, 100, 100),
            [-1.0, 1.0, 1.0, NAN],
            short_only=True,
            order_quantity=2.0,
        )
        eq(r.order_size, [-2, 2, NAN, NAN])
        eq(r.position, [0, -2, 0, 0])

    def test_short_only_never_opens_a_long(self):
        r = run(flat(100, 100, 100), [1.0, 1.0, NAN], short_only=True)
        eq(r.position, [0, 0, 0])


# long 1 @ 100, filled at bar 1 (bar 0 places the order at its close)
ENTRY: list[Bar] = [(100, 100, 100, 100), (100, 100, 100, 100)]


class TestStopLoss:
    def test_non_immediate_fills_at_stop_price(self):
        r = run([*ENTRY, (100, 101, 89, 95)], [1.0, NAN, NAN], stop_loss=0.1)
        eq(r.trade_price, [NAN, 100, 90])
        eq(r.trade_size, [NAN, 1, -1])
        eq(r.position, [0, 1, 0])
        eq(r.avg_price, [NAN, 100, NAN])
        # recorded on the order row (the bar the stop was placed)
        eq(r.stop_price, [NAN, 90, NAN])

    def test_untouched_stop_does_nothing(self):
        r = run([*ENTRY, (100, 101, 91, 95)], [1.0, NAN, NAN], stop_loss=0.1)
        eq(r.position, [0, 1, 1])
        eq(r.stop_price, [NAN, NAN, NAN])

    @pytest.mark.parametrize(
        ("fill_at_open", "expected"), [(False, 88.0), (True, 87.0)]
    )
    def test_immediate_fills_per_exit_policy(self, fill_at_open: bool, expected: float):
        # entry bar itself closes below the stop → stop is breached when placed
        r = run(
            [(100, 100, 100, 100), (100, 100, 85, 88), (87, 89, 86, 88)],
            [1.0, NAN, NAN],
            stop_loss=0.1,
            exit_market_fill_at_open=fill_at_open,
        )
        eq(r.trade_price, [NAN, 100, expected])
        eq(r.position, [0, 1, 0])
        eq(r.stop_price, [NAN, 90, NAN])

    @pytest.mark.parametrize("fill_at_open", [False, True])
    def test_gap_through_fills_at_open(self, fill_at_open: bool):
        r = run(
            [*ENTRY, (85, 86, 84, 85)],
            [1.0, NAN, NAN],
            stop_loss=0.1,
            exit_market_fill_at_open=fill_at_open,
        )
        eq(r.trade_price, [NAN, 100, 85])
        eq(r.stop_price, [NAN, 90, NAN])

    def test_short_position_stop_is_above(self):
        r = run([*ENTRY, (100, 111, 99, 100)], [-1.0, NAN, NAN], stop_loss=0.1)
        eq(r.trade_price, [NAN, 100, 110])
        eq(r.trade_size, [NAN, -1, 1])
        eq(r.position, [0, -1, 0])

    def test_short_gap_through_fills_at_open(self):
        r = run([*ENTRY, (115, 116, 114, 115)], [-1.0, NAN, NAN], stop_loss=0.1)
        eq(r.trade_price, [NAN, 100, 115])

    def test_stop_is_recomputed_from_the_new_avg_price(self):
        r = run(
            [*ENTRY, (120, 120, 120, 120), (120, 120, 98, 100)],
            [1.0, 1.0, NAN, NAN],
            stop_loss=0.1,
            market_fill_at_open=True,
        )
        eq(r.avg_price, [NAN, 100, 110, NAN])
        eq(r.trade_price, [NAN, 100, 120, 99])
        eq(r.trade_size, [NAN, 1, 1, -2])
        eq(r.stop_price, [NAN, NAN, 99, NAN])

    def test_stop_value_is_read_at_the_placement_bar(self):
        r = run(
            [*ENTRY, (100, 100, 94, 96), (96, 96, 96, 96)],
            [1.0, NAN, NAN, NAN],
            stop_loss=[NAN, 0.05, NAN, NAN],
        )
        eq(r.trade_price, [NAN, 100, 95, NAN])

    def test_disabled_stop_never_fires(self):
        r = run([*ENTRY, (100, 100, 1, 1)], [1.0, NAN, NAN])
        eq(r.position, [0, 1, 1])


class TestTakeProfit:
    def test_non_immediate_fills_at_trigger_price(self):
        r = run([*ENTRY, (100, 112, 99, 105)], [1.0, NAN, NAN], take_profit=0.1)
        eq(r.trade_price, [NAN, 100, 110])
        eq(r.position, [0, 1, 0])
        eq(r.stop_price, [NAN, 110, NAN])

    @pytest.mark.parametrize(
        ("fill_at_open", "expected"), [(False, 112.0), (True, 113.0)]
    )
    def test_immediate_fills_per_exit_policy(self, fill_at_open: bool, expected: float):
        r = run(
            [(100, 100, 100, 100), (100, 115, 100, 112), (113, 114, 112, 113)],
            [1.0, NAN, NAN],
            take_profit=0.1,
            exit_market_fill_at_open=fill_at_open,
        )
        eq(r.trade_price, [NAN, 100, expected])

    def test_gap_through_fills_at_open(self):
        r = run([*ENTRY, (115, 116, 114, 115)], [1.0, NAN, NAN], take_profit=0.1)
        eq(r.trade_price, [NAN, 100, 115])

    def test_short_position_take_profit_is_below(self):
        r = run([*ENTRY, (100, 101, 88, 95)], [-1.0, NAN, NAN], take_profit=0.1)
        eq(r.trade_price, [NAN, 100, 90])
        eq(r.trade_size, [NAN, -1, 1])

    def test_stop_loss_wins_when_both_trigger_inside_the_bar(self):
        r = run(
            [*ENTRY, (100, 112, 89, 100)],
            [1.0, NAN, NAN],
            stop_loss=0.1,
            take_profit=0.1,
        )
        eq(r.trade_price, [NAN, 100, 90])
        eq(r.stop_price, [NAN, 90, NAN])

    def test_gap_through_stop_loss_wins_over_immediate_take_profit(self):
        # prev close 112 breaches TP 110, then the open gaps below SL 90
        r = run(
            [(100, 100, 100, 100), (100, 115, 100, 112), (85, 86, 84, 85)],
            [1.0, NAN, NAN],
            stop_loss=0.1,
            take_profit=0.1,
        )
        # SL gap-through outranks the immediate TP → fills at open, not prev close
        eq(r.trade_price, [NAN, 100, 85])
        eq(r.stop_price, [NAN, 90, NAN])

    def test_immediate_take_profit_beats_non_immediate_stop_loss(self):
        # prev close 112 >= TP 110 → TP is already triggered before bar 2 trades;
        # the bar later crashes through SL 90, which must not rewrite the exit
        r = run(
            [(100, 100, 100, 100), (100, 115, 100, 112), (112, 113, 85, 90)],
            [1.0, NAN, NAN],
            stop_loss=0.1,
            take_profit=0.1,
        )
        eq(r.trade_price, [NAN, 100, 112])
        eq(r.stop_price, [NAN, 110, NAN])

    def test_gap_through_take_profit_beats_non_immediate_stop_loss(self):
        # open 115 gaps above TP 110 → exits at the open, before the crash to SL 90
        r = run(
            [*ENTRY, (115, 116, 85, 90)],
            [1.0, NAN, NAN],
            stop_loss=0.1,
            take_profit=0.1,
        )
        eq(r.trade_price, [NAN, 100, 115])
        eq(r.stop_price, [NAN, 110, NAN])

    def test_immediate_take_profit_beats_non_immediate_trailing_stop(self):
        # best 112 → trailing stop 109.76; TP 110; prev close 111 breaches TP only
        r = run(
            [(100, 100, 100, 100), (100, 112, 100, 111), (111, 111, 109, 109)],
            [1.0, NAN, NAN],
            trailing_stop=0.02,
            take_profit=0.1,
        )
        eq(r.trade_price, [NAN, 100, 111])


class TestTrailingStop:
    def test_trails_the_best_price_since_entry(self):
        r = run(
            [*ENTRY, (100, 120, 100, 120), (120, 121, 107, 110)],
            [1.0, NAN, NAN, NAN],
            trailing_stop=0.1,
        )
        eq(r.trade_price, [NAN, 100, NAN, 108])
        eq(r.stop_price, [NAN, NAN, 108, NAN])
        eq(r.position, [0, 1, 1, 0])

    def test_does_not_loosen_when_price_falls(self):
        r = run(
            [*ENTRY, (100, 120, 100, 120), (120, 120, 110, 110), (110, 110, 107, 108)],
            [1.0, NAN, NAN, NAN, NAN],
            trailing_stop=0.1,
        )
        eq(r.trade_price, [NAN, 100, NAN, NAN, 108])

    def test_replaces_stop_loss_when_both_set(self):
        r = run(
            [*ENTRY, (100, 120, 100, 120), (120, 121, 107, 110)],
            [1.0, NAN, NAN, NAN],
            stop_loss=0.5,
            trailing_stop=0.1,
        )
        eq(r.trade_price, [NAN, 100, NAN, 108])

    def test_short_trails_the_lowest_low(self):
        r = run(
            [*ENTRY, (100, 100, 80, 80), (80, 89, 79, 85)],
            [-1.0, NAN, NAN, NAN],
            trailing_stop=0.1,
        )
        eq(r.trade_price, [NAN, 100, NAN, 88])
        eq(r.trade_size, [NAN, -1, NAN, 1])

    def test_best_price_resets_on_a_new_position(self):
        bars: list[Bar] = [
            *ENTRY,
            (100, 200, 100, 200),  # best = 200 → stop 180
            (200, 200, 170, 170),  # stopped out at 180
            (170, 170, 170, 170),  # re-entry order placed (signal 1)
            (170, 170, 170, 170),  # filled at 170 → best = 170, stop 153
            (170, 170, 160, 160),  # 160 > 153 → must NOT stop out
        ]
        r = run(bars, [1.0, NAN, NAN, NAN, 1.0, NAN, NAN], trailing_stop=0.1)
        eq(r.trade_price, [NAN, 100, NAN, 180, NAN, 170, NAN])
        eq(r.position, [0, 1, 1, 0, 0, 1, 1])

    def test_best_price_resets_on_a_flip(self):
        bars: list[Bar] = [
            *ENTRY,
            (100, 200, 100, 200),  # long best = 200; flip order placed at close 200
            (200, 200, 200, 200),  # flipped short @ 200 → best = 200, stop 220
            (200, 215, 200, 210),  # 215 < 220 → must NOT stop out
        ]
        r = run(bars, [1.0, NAN, -1.0, NAN, NAN], trailing_stop=0.1)
        eq(r.position, [0, 1, 1, -1, -1])


class TestTimeWindow:
    @pytest.mark.parametrize(
        ("fill_at_open", "expected"), [(False, 105.0), (True, 107.0)]
    )
    def test_closes_after_n_bars_per_exit_policy(
        self, fill_at_open: bool, expected: float
    ):
        r = run(
            [*ENTRY, (105, 105, 105, 105), (107, 110, 107, 110), (110, 110, 110, 110)],
            [1.0, NAN, NAN, NAN, NAN],
            time_window=2,
            exit_market_fill_at_open=fill_at_open,
        )
        eq(r.trade_price, [NAN, 100, NAN, expected, NAN])
        eq(r.trade_size, [NAN, 1, NAN, -1, NAN])
        eq(r.position, [0, 1, 1, 0, 0])
        eq(r.stop_price, [NAN] * 5)

    def test_window_of_one_closes_on_the_bar_after_entry(self):
        r = run(flat(100, 100, 100, 100), [1.0, NAN, NAN, NAN], time_window=1)
        eq(r.position, [0, 1, 0, 0])

    def test_scaling_in_does_not_restart_the_window(self):
        r = run(flat(100, 100, 100, 100, 100), [1.0, 1.0, NAN, NAN, NAN], time_window=2)
        eq(r.position, [0, 1, 2, 0, 0])
        eq(r.trade_size, [NAN, 1, 1, -2, NAN])

    def test_flip_restarts_the_window(self):
        r = run(flat(*[100] * 7), [1.0, NAN, -1.0, NAN, NAN, NAN, NAN], time_window=3)
        eq(r.position, [0, 1, 1, -1, -1, -1, 0])

    def test_reentry_restarts_the_window(self):
        r = run(flat(*[100] * 7), [1.0, NAN, 1.0, NAN, NAN, NAN, NAN], time_window=1)
        # closed at bar 2; bar 2's order re-opens at bar 3; closed again at bar 4
        eq(r.position, [0, 1, 0, 1, 0, 0, 0])

    def test_disabled_window_holds_forever(self):
        r = run(flat(*[100] * 6), [1.0, NAN, NAN, NAN, NAN, NAN], time_window=-1)
        eq(r.position, [0, 1, 1, 1, 1, 1])


class TestFillPriority:
    def test_immediate_stop_cancels_the_pending_market_order(self):
        r = run(
            [
                (100, 100, 100, 100),
                (100, 100, 85, 88),
                (88, 88, 88, 88),
                (88, 88, 88, 88),
            ],
            [1.0, 1.0, NAN, NAN],
            stop_loss=0.1,
        )
        eq(r.order_size, [1, 1, NAN, NAN])
        eq(r.trade_size, [NAN, 1, -1, NAN])
        eq(r.position, [0, 1, 0, 0])

    def test_time_window_close_cancels_the_pending_market_order(self):
        r = run(flat(*[100] * 5), [1.0, NAN, -1.0, NAN, NAN], time_window=2)
        eq(r.order_size, [1, NAN, -2, NAN, NAN])
        eq(r.trade_size, [NAN, 1, NAN, -1, NAN])
        eq(r.position, [0, 1, 1, 0, 0])

    def test_immediate_stop_beats_time_window_close(self):
        r = run(
            [(100, 100, 100, 100), (100, 100, 85, 88), (87, 87, 87, 87)],
            [1.0, NAN, NAN],
            stop_loss=0.1,
            time_window=1,
        )
        eq(r.stop_price, [NAN, 90, NAN])
        eq(r.position, [0, 1, 0])

    def test_market_order_beats_non_immediate_stop(self):
        r = run([*ENTRY, (100, 100, 85, 95)], [1.0, 1.0, NAN], stop_loss=0.1)
        eq(r.trade_price, [NAN, 100, 100])
        eq(r.position, [0, 1, 2])
        eq(r.stop_price, [NAN, NAN, NAN])

    def test_limit_closer_to_open_fills_before_the_stop(self):
        r = run(
            [*ENTRY, (100, 100, 85, 90)],
            [1.0, 1.0, NAN],
            order_price=[100.0, 95.0, NAN],
            stop_loss=0.1,
        )
        eq(r.trade_price, [NAN, 100, 95])
        eq(r.position, [0, 1, 2])
        eq(r.avg_price, [NAN, 100, 97.5])
        eq(r.stop_price, [NAN, NAN, NAN])

    def test_stop_fills_before_a_limit_beyond_it(self):
        r = run(
            [*ENTRY, (100, 100, 85, 90)],
            [1.0, 1.0, NAN],
            order_price=[100.0, 88.0, NAN],
            stop_loss=0.1,
        )
        eq(r.trade_price, [NAN, 100, 90])
        eq(r.position, [0, 1, 0])
        eq(r.stop_price, [NAN, 90, NAN])

    def test_sell_limit_fills_before_the_take_profit_above_it(self):
        r = run(
            [*ENTRY, (100, 112, 100, 110)],
            [1.0, -1.0, NAN],
            order_price=[100.0, 105.0, NAN],
            take_profit=0.1,
        )
        eq(r.trade_price, [NAN, 100, 105])
        eq(r.trade_size, [NAN, 1, -2])
        eq(r.position, [0, 1, -1])
        eq(r.avg_price, [NAN, 100, 105])

    def test_at_most_one_trade_per_bar(self):
        # limit add fills first; the stop the bar later breaches waits for bar 3
        r = run(
            [*ENTRY, (100, 100, 85, 90), (90, 90, 90, 90)],
            [1.0, 1.0, NAN, NAN],
            order_price=[100.0, 95.0, NAN, NAN],
            stop_loss=0.05,
        )
        eq(r.trade_size, [NAN, 1, 1, -2])
        # avg 97.5 → stop 92.625, already breached by bar 2's close → immediate
        eq(r.stop_price, [NAN, NAN, 92.625, NAN])
        eq(r.trade_price, [NAN, 100, 95, 90])


class TestUniverse:
    def test_leaving_the_universe_closes_at_that_bars_close(self):
        r = run(
            [*ENTRY, (103, 106, 102, 105), (110, 110, 110, 110)],
            [1.0, NAN, NAN, NAN],
            in_universe=[True, True, False, False],
        )
        eq(r.trade_price, [NAN, 100, 105, NAN])
        eq(r.trade_size, [NAN, 1, -1, NAN])
        eq(r.position, [0, 1, 0, 0])
        eq(r.avg_price, [NAN, 100, NAN, NAN])

    def test_pending_order_is_cancelled_not_filled(self):
        r = run(flat(100, 100, 100), [1.0, NAN, NAN], in_universe=[True, False, True])
        eq(r.order_size, [1, NAN, NAN])
        eq(r.trade_size, [NAN, NAN, NAN])
        eq(r.position, [0, 0, 0])

    def test_pending_stop_is_cancelled_and_exit_is_at_close(self):
        r = run(
            [*ENTRY, (100, 100, 50, 60)],
            [1.0, NAN, NAN],
            in_universe=[True, True, False],
            stop_loss=0.1,
        )
        eq(r.trade_price, [NAN, 100, 60])
        eq(r.stop_price, [NAN, NAN, NAN])

    def test_short_position_is_closed_too(self):
        r = run(flat(100, 100, 90), [-1.0, NAN, NAN], in_universe=[True, True, False])
        eq(r.trade_price, [NAN, 100, 90])
        eq(r.trade_size, [NAN, -1, 1])

    def test_position_carries_across_covered_bars(self):
        r = run(flat(*[100] * 5), [1.0, NAN, NAN, NAN, NAN], in_universe=True)
        eq(r.position, [0, 1, 1, 1, 1])

    def test_reentry_after_a_gap(self):
        r = run(
            flat(*[100] * 5),
            [1.0, NAN, NAN, 1.0, NAN],
            in_universe=[True, True, False, True, True],
        )
        eq(r.position, [0, 1, 0, 0, 1])

    def test_trailing_stop_and_time_window_restart_after_a_gap(self):
        bars: list[Bar] = [
            *ENTRY,
            (100, 300, 100, 300),  # best = 300 while in the first segment
            (300, 300, 300, 300),  # gap → closed at 300
            (100, 100, 100, 100),  # re-entry order
            (100, 100, 100, 100),  # filled @ 100 → best must restart at 100
            (100, 100, 95, 95),  # 95 > 90 → must NOT stop out
        ]
        r = run(
            bars,
            [1.0, NAN, NAN, NAN, 1.0, NAN, NAN],
            in_universe=[True, True, True, False, True, True, True],
            trailing_stop=0.1,
            time_window=5,
        )
        eq(r.position, [0, 1, 1, 0, 0, 1, 1])


def _random_case(seed: int, n: int = 150) -> tuple[dict[str, Any], Result]:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.03, n)))
    open_ = np.empty(n)
    open_[0] = close[0]
    # gappy opens so gap-through paths are exercised
    open_[1:] = close[:-1] * np.exp(rng.normal(0.0, 0.02, n - 1))
    high = np.maximum(open_, close) * (1.0 + rng.uniform(0.0, 0.03, n))
    low = np.minimum(open_, close) * (1.0 - rng.uniform(0.0, 0.03, n))
    bars = list(
        zip(open_.tolist(), high.tolist(), low.tolist(), close.tolist(), strict=True)
    )

    def maybe(value: float) -> float:
        return value if rng.random() < 0.5 else NAN

    direction = rng.choice(["both", "long", "short"])
    in_universe = np.ones(n, dtype=np.bool_)
    if rng.random() < 0.5:
        start = int(rng.integers(1, n - 10))
        in_universe[start : start + int(rng.integers(1, 10))] = False
    kwargs: dict[str, Any] = {
        "signal": np.where(
            in_universe, rng.choice([1.0, -1.0, NAN], size=n, p=[0.2, 0.2, 0.6]), NAN
        ),
        "in_universe": in_universe,
        # a mix of market orders (at close) and limits around the close
        "order_price": close
        * np.where(rng.random(n) < 0.5, 1.0, 1.0 + rng.normal(0.0, 0.02, n)),
        "order_quantity": rng.integers(1, 4, n).astype(np.float64),
        "first_only": bool(rng.random() < 0.5),
        "long_only": direction == "long",
        "short_only": direction == "short",
        "take_profit": maybe(float(rng.uniform(0.01, 0.1))),
        "stop_loss": maybe(float(rng.uniform(0.01, 0.1))),
        "trailing_stop": maybe(float(rng.uniform(0.01, 0.1))),
        "time_window": int(rng.integers(1, 8)) if rng.random() < 0.5 else -1,
        "market_fill_at_open": bool(rng.random() < 0.5),
        "exit_market_fill_at_open": bool(rng.random() < 0.5),
    }
    return {"bars": bars, **kwargs}, run(bars, **kwargs)


@pytest.mark.parametrize("seed", range(200))
class TestInvariants:
    """Properties that must hold for any input, checked on random paths."""

    def test_position_is_the_running_sum_of_trades(self, seed: int):
        _, r = _random_case(seed)
        np.testing.assert_allclose(
            r.position, np.cumsum(np.nan_to_num(r.trade_size)), atol=1e-9
        )

    def test_trade_price_and_size_come_together(self, seed: int):
        _, r = _random_case(seed)
        np.testing.assert_array_equal(np.isnan(r.trade_price), np.isnan(r.trade_size))
        np.testing.assert_array_equal(np.isnan(r.order_price), np.isnan(r.order_size))
        assert np.isnan(r.trade_size[0])
        assert not np.any(r.trade_size == 0.0)

    def test_avg_price_exists_exactly_while_in_position(self, seed: int):
        _, r = _random_case(seed)
        np.testing.assert_array_equal(np.isnan(r.avg_price), r.position == 0.0)
        assert np.all(r.avg_price[r.position != 0.0] > 0.0)

    def test_trades_fill_at_a_price_the_bar_could_trade(self, seed: int):
        case, r = _random_case(seed)
        ohlc = np.asarray(case["bars"])
        high, low, close = ohlc[:, 1], ohlc[:, 2], ohlc[:, 3]
        for i in np.flatnonzero(~np.isnan(r.trade_price)):
            px = r.trade_price[i]
            in_bar = low[i] - 1e-9 <= px <= high[i] + 1e-9
            assert in_bar or px == close[i - 1], (
                f"bar {i}: {px} outside [{low[i]}, {high[i]}]"
            )

    def test_every_trade_has_a_cause_on_the_previous_bar(self, seed: int):
        case, r = _random_case(seed)
        for i in np.flatnonzero(~np.isnan(r.trade_size)):
            had_order = not np.isnan(r.order_size[i - 1])
            had_position = r.position[i - 1] != 0.0
            assert had_order or had_position
            if not case["in_universe"][i]:
                assert r.position[i] == 0.0

    def test_flat_on_every_out_of_universe_bar(self, seed: int):
        case, r = _random_case(seed)
        assert np.all(r.position[~case["in_universe"]] == 0.0)

    def test_direction_constraints_hold(self, seed: int):
        case, r = _random_case(seed)
        if case["long_only"]:
            assert np.all(r.position >= 0.0)
        if case["short_only"]:
            assert np.all(r.position <= 0.0)

    def test_a_recorded_stop_closes_the_whole_position_next_bar(self, seed: int):
        _, r = _random_case(seed)
        for i in np.flatnonzero(~np.isnan(r.stop_price)):
            assert r.position[i] != 0.0
            assert r.trade_size[i + 1] == -r.position[i]
            assert r.position[i + 1] == 0.0

    def test_a_stop_fills_at_its_price_the_open_or_the_prev_close(self, seed: int):
        case, r = _random_case(seed)
        ohlc = np.asarray(case["bars"])
        open_, close = ohlc[:, 0], ohlc[:, 3]
        for i in np.flatnonzero(~np.isnan(r.stop_price)):
            px = r.trade_price[i + 1]
            assert px in (r.stop_price[i], open_[i + 1], close[i])
