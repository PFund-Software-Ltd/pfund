from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import pytest

from pfund.backtest.portfolio_backtest_kernel import portfolio_backtest_loop_kernel

NAN = np.nan
CAPITAL = 1000.0
A, B, C = 0, 1, 2  # product columns

type FloatArray = np.ndarray[Any, np.dtype[np.float64]]
type BoolArray = np.ndarray[Any, np.dtype[np.bool_]]


class Result(NamedTuple):
    order_price: FloatArray  # [T, P]
    order_size: FloatArray  # [T, P]
    trade_price: FloatArray  # [T, P]
    trade_size: FloatArray  # [T, P]
    position: FloatArray  # [T, P]
    avg_price: FloatArray  # [T, P]
    cash: FloatArray  # [T]
    equity: FloatArray  # [T]


def run(
    prices: Any,
    weights: Any,
    *,
    in_universe: Any = None,  # None = wherever the product has a price
    initial_capital: float = CAPITAL,
    compound: bool = True,
) -> Result:
    """Run the kernel on per-product series: prices[j] / weights[j] = product j."""
    price_mat = np.ascontiguousarray(np.asarray(prices, dtype=np.float64).T)
    weight_mat = np.ascontiguousarray(np.asarray(weights, dtype=np.float64).T)
    assert price_mat.shape == weight_mat.shape
    if in_universe is None:
        universe = ~np.isnan(price_mat)
    else:
        universe = np.ascontiguousarray(np.asarray(in_universe, dtype=np.bool_).T)
        assert universe.shape == price_mat.shape
    T, P = price_mat.shape
    return Result(
        *portfolio_backtest_loop_kernel(
            price_mat, weight_mat, universe, initial_capital, compound, T, P
        )
    )


def eq(actual: FloatArray, expected: list[float]) -> None:
    np.testing.assert_allclose(
        actual, np.asarray(expected, dtype=np.float64), equal_nan=True
    )


class TestBasics:
    def test_no_weight_produces_no_orders_or_trades(self):
        r = run([[100, 101, 102]], [[NAN, NAN, NAN]])
        for out in (r.order_price, r.order_size, r.trade_price, r.trade_size):
            eq(out[:, A], [NAN, NAN, NAN])
        eq(r.position[:, A], [0, 0, 0])
        eq(r.avg_price[:, A], [NAN, NAN, NAN])
        eq(r.cash, [CAPITAL, CAPITAL, CAPITAL])
        eq(r.equity, [CAPITAL, CAPITAL, CAPITAL])

    def test_single_date(self):
        r = run([[100]], [[1.0]])
        eq(r.order_size[:, A], [10])
        eq(r.trade_size[:, A], [NAN])
        eq(r.position[:, A], [0])
        eq(r.equity, [CAPITAL])

    def test_order_on_last_date_never_fills(self):
        r = run([[100, 100, 100]], [[NAN, NAN, 1.0]])
        eq(r.order_size[:, A], [NAN, NAN, 10])
        eq(r.trade_size[:, A], [NAN, NAN, NAN])
        eq(r.position[:, A], [0, 0, 0])

    def test_output_shapes_and_dtypes(self):
        r = run([[100, 100, 100, 100], [50, 50, 50, 50]], [[1.0, NAN, NAN, NAN]] * 2)
        for out in r[:6]:
            assert out.dtype == np.float64
            assert out.shape == (4, 2)
        for out in (r.cash, r.equity):
            assert out.dtype == np.float64
            assert out.shape == (4,)


class TestRebalancing:
    def test_fills_next_date_at_prev_close(self):
        r = run([[100, 110, 120]], [[1.0, NAN, NAN]])
        eq(r.order_price[:, A], [100, NAN, NAN])
        eq(r.order_size[:, A], [10, NAN, NAN])
        eq(r.trade_price[:, A], [NAN, 100, NAN])
        eq(r.trade_size[:, A], [NAN, 10, NAN])
        eq(r.position[:, A], [0, 10, 10])
        eq(r.avg_price[:, A], [NAN, 100, 100])
        eq(r.cash, [1000, 0, 0])
        eq(r.equity, [1000, 1100, 1200])

    def test_weights_split_equity_across_products(self):
        r = run([[100, 100], [50, 50]], [[0.6, NAN], [0.4, NAN]])
        eq(r.order_size[0], [6, 8])
        eq(r.position[1], [6, 8])
        eq(r.cash, [1000, 0])

    def test_weights_below_one_leave_cash(self):
        r = run([[100, 100]], [[0.3, NAN]])
        eq(r.position[:, A], [0, 3])
        eq(r.cash, [1000, 700])
        eq(r.equity, [1000, 1000])

    def test_already_on_target_places_no_order(self):
        r = run([[100, 100, 100]], [[1.0, 1.0, 1.0]])
        eq(r.order_size[:, A], [10, NAN, NAN])
        eq(r.trade_size[:, A], [NAN, 10, NAN])

    def test_rebalances_drifted_positions_back_to_target(self):
        # A doubles: 50/50 becomes 1000 vs 500 of a 1500 pot → back to 750 each
        r = run([[100, 200, 200], [100, 100, 100]], [[0.5, 0.5, NAN]] * 2)
        eq(r.equity, [1000, 1500, 1500])
        eq(r.order_size[1], [-1.25, 2.5])
        eq(r.order_price[1], [200, 100])
        eq(r.position[2], [3.75, 7.5])
        eq(r.cash, [1000, 0, 0])

    def test_zero_weight_closes_the_position(self):
        r = run([[100, 110, 110]], [[1.0, 0.0, NAN]])
        eq(r.order_size[:, A], [10, -10, NAN])
        eq(r.trade_price[:, A], [NAN, 100, 110])
        eq(r.position[:, A], [0, 10, 0])
        eq(r.avg_price[:, A], [NAN, 100, NAN])
        eq(r.cash, [1000, 0, 1100])
        eq(r.equity, [1000, 1100, 1100])

    def test_zero_weight_on_a_flat_product_places_no_order(self):
        r = run([[100, 100]], [[0.0, NAN]])
        eq(r.order_size[:, A], [NAN, NAN])
        eq(r.position[:, A], [0, 0])

    def test_negative_weight_opens_a_short(self):
        r = run([[100, 100, 90]], [[-0.5, NAN, NAN]])
        eq(r.order_size[:, A], [-5, NAN, NAN])
        eq(r.position[:, A], [0, -5, -5])
        eq(r.avg_price[:, A], [NAN, 100, 100])
        eq(r.cash, [1000, 1500, 1500])
        eq(r.equity, [1000, 1000, 1050])

    def test_weights_above_one_borrow_cash(self):
        r = run([[100, 110]], [[2.0, NAN]])
        eq(r.position[:, A], [0, 20])
        eq(r.cash, [1000, -1000])
        eq(r.equity, [1000, 1200])

    def test_order_price_and_size_are_frozen_at_placement(self):
        # sized and priced off close[0]; the move to 150 changes neither
        r = run([[100, 150]], [[1.0, NAN]])
        eq(r.trade_price[:, A], [NAN, 100])
        eq(r.trade_size[:, A], [NAN, 10])
        eq(r.equity, [1000, 1500])

    def test_product_order_does_not_change_the_result(self):
        prices = [[100, 200, 150], [50, 40, 60], [10, 10, 12]]
        weights = [[0.5, 0.2, NAN], [0.3, NAN, NAN], [0.2, 0.5, NAN]]
        r = run(prices, weights)
        flipped = run(prices[::-1], weights[::-1])
        for out, out_flipped in zip(r[:6], flipped[:6], strict=True):
            np.testing.assert_allclose(out, out_flipped[:, ::-1], equal_nan=True)
        eq(r.cash, flipped.cash.tolist())
        eq(r.equity, flipped.equity.tolist())


class TestCompound:
    def test_compound_sizes_on_current_equity(self):
        # equity doubled with the price: 100% of 2000 at 200 is still 10 units
        r = run([[100, 200, 200]], [[1.0, 1.0, NAN]], compound=True)
        eq(r.order_size[:, A], [10, NAN, NAN])
        eq(r.position[:, A], [0, 10, 10])

    def test_non_compound_sizes_on_initial_capital(self):
        # 100% of the initial 1000 at 200 is 5 units: the gain is taken as cash
        r = run([[100, 200, 200]], [[1.0, 1.0, NAN]], compound=False)
        eq(r.order_size[:, A], [10, -5, NAN])
        eq(r.position[:, A], [0, 10, 5])
        eq(r.cash, [1000, 0, 1000])
        eq(r.equity, [1000, 2000, 2000])

    def test_non_compound_ignores_losses_too(self):
        r = run([[100, 50, 50]], [[1.0, 1.0, NAN]], compound=False)
        eq(r.order_size[:, A], [10, 10, NAN])
        eq(r.position[:, A], [0, 10, 20])
        eq(r.cash, [1000, 0, -500])
        eq(r.equity, [1000, 500, 500])


class TestAvgPrice:
    def test_adding_averages_the_price(self):
        r = run([[100, 50, 50]], [[0.5, 1.0, NAN]])
        eq(r.trade_size[:, A], [NAN, 5, 10])
        eq(r.trade_price[:, A], [NAN, 100, 50])
        eq(r.position[:, A], [0, 5, 15])
        eq(r.avg_price[:, A], [NAN, 100, 1000 / 15])

    def test_partial_reduction_keeps_the_avg_price(self):
        r = run([[100, 200, 200]], [[1.0, 0.5, NAN]])
        eq(r.trade_size[:, A], [NAN, 10, -5])
        eq(r.position[:, A], [0, 10, 5])
        eq(r.avg_price[:, A], [NAN, 100, 100])
        eq(r.cash, [1000, 0, 1000])

    def test_flip_restarts_the_avg_price_at_the_fill(self):
        r = run([[100, 200, 200]], [[1.0, -1.0, NAN]])
        eq(r.order_size[:, A], [10, -20, NAN])
        eq(r.position[:, A], [0, 10, -10])
        eq(r.avg_price[:, A], [NAN, 100, 200])
        eq(r.cash, [1000, 0, 4000])
        eq(r.equity, [1000, 2000, 2000])

    def test_short_avg_price_stays_positive(self):
        r = run([[100, 80, 80]], [[-0.5, -1.0, NAN]])
        # t1: equity 1100, target -13.75 → sell 8.75 more at 80
        eq(r.trade_size[:, A], [NAN, -5, -8.75])
        eq(r.avg_price[:, A], [NAN, 100, 1200 / 13.75])

    def test_reopening_after_a_close_starts_fresh(self):
        r = run([[100, 120, 150, 150]], [[1.0, 0.0, 1.0, NAN]])
        eq(r.position[:, A], [0, 10, 0, 8])
        eq(r.avg_price[:, A], [NAN, 100, NAN, 150])


class TestFreeCapital:
    def test_weights_are_fractions_of_capital_not_held_by_kept_positions(self):
        # A is kept (nan) while worth 1000 of the 1500 pot: B and C split the
        # other 500, so kept + explicit never exceeds the pot
        r = run(
            [[100, 200, 200], [100, 100, 100], [100, 100, 100]],
            [[0.5, NAN, NAN], [NAN, 0.5, NAN], [NAN, 0.5, NAN]],
        )
        eq(r.equity, [1000, 1500, 1500])
        eq(r.order_size[1], [NAN, 2.5, 2.5])
        eq(r.position[2], [5, 2.5, 2.5])
        eq(r.cash, [1000, 500, 0])

    def test_non_compound_reserves_kept_positions_from_initial_capital(self):
        r = run(
            [[100, 150, 150], [100, 100, 100]],
            [[0.5, NAN, NAN], [NAN, 1.0, NAN]],
            compound=False,
        )
        # free = 1000 - 5 * 150
        eq(r.order_size[1], [NAN, 2.5])

    def test_a_kept_short_adds_to_free_capital(self):
        r = run([[100, 100, 100], [100, 100, 100]], [[-0.5, NAN, NAN], [NAN, 1.0, NAN]])
        # free = 1000 - (-5 * 100)
        eq(r.order_size[1], [NAN, 15])
        eq(r.cash, [1000, 1500, 0])
        eq(r.equity, [1000, 1000, 1000])

    def test_raises_when_kept_positions_use_all_the_capital(self):
        with pytest.raises(ValueError, match="free capital"):
            run([[100, 100], [100, 100]], [[1.0, NAN], [NAN, 0.5]])

    def test_raises_when_non_compound_kept_position_outgrows_initial_capital(self):
        with pytest.raises(ValueError, match="free capital"):
            run([[100, 300], [100, 100]], [[0.5, NAN], [NAN, 0.5]], compound=False)

    def test_holding_everything_never_raises(self):
        r = run([[100, 100, 100]], [[1.0, NAN, NAN]])
        eq(r.order_size[:, A], [10, NAN, NAN])
        eq(r.position[:, A], [0, 10, 10])

    def test_zero_weight_never_raises(self):
        # B's close needs no sizing, so no free capital is fine
        r = run(
            [[100, 200, 200], [100, 100, 100]],
            [[0.5, NAN, NAN], [0.5, 0.0, NAN]],
            compound=False,
        )
        eq(r.order_size[1], [NAN, -5])
        eq(r.position[2], [5, 0])

    def test_weight_outside_the_universe_never_raises(self):
        # B's weight is ignored for sizing, so it must not count as a rebalance
        r = run(
            [[100, 110, 120], [50, 50, 50]],
            [[1.0, NAN, NAN], [NAN, 0.5, NAN]],
            in_universe=[[True, True, True], [True, False, True]],
        )
        eq(r.order_size[:, B], [NAN, NAN, NAN])
        eq(r.position[:, A], [0, 10, 10])
        eq(r.equity, [1000, 1100, 1200])


class TestUniverse:
    def test_leaving_the_universe_closes_at_that_dates_close(self):
        r = run(
            [[100, 110, 120, 130]],
            [[1.0, NAN, NAN, NAN]],
            in_universe=[[True, True, False, True]],
        )
        eq(r.order_price[:, A], [100, NAN, 120, NAN])
        eq(r.order_size[:, A], [10, NAN, -10, NAN])
        eq(r.trade_price[:, A], [NAN, 100, NAN, 120])
        eq(r.trade_size[:, A], [NAN, 10, NAN, -10])
        eq(r.position[:, A], [0, 10, 10, 0])
        eq(r.avg_price[:, A], [NAN, 100, 100, NAN])
        eq(r.cash, [1000, 0, 0, 1200])
        eq(r.equity, [1000, 1100, 1200, 1200])

    def test_committed_order_still_fills_after_leaving(self):
        r = run(
            [[100, 110, 120]], [[1.0, NAN, NAN]], in_universe=[[True, False, False]]
        )
        eq(r.trade_size[:, A], [NAN, 10, -10])
        eq(r.trade_price[:, A], [NAN, 100, 110])
        eq(r.position[:, A], [0, 10, 0])
        eq(r.cash, [1000, 0, 1100])

    def test_short_position_is_closed_too(self):
        r = run(
            [[100, 100, 90, 90]],
            [[-0.5, NAN, NAN, NAN]],
            in_universe=[[True, True, False, False]],
        )
        eq(r.order_size[:, A], [-5, NAN, 5, NAN])
        eq(r.position[:, A], [0, -5, -5, 0])
        eq(r.cash, [1000, 1500, 1500, 1050])

    def test_weight_outside_the_universe_is_ignored(self):
        r = run([[100, 100]], [[1.0, NAN]], in_universe=[[False, True]])
        eq(r.order_size[:, A], [NAN, NAN])
        eq(r.position[:, A], [0, 0])

    def test_leaving_positions_capital_is_free_for_new_weights(self):
        # A leaves on the date B is bought: B is sized on the full pot, and A's
        # close pays for it on the same fill date
        r = run(
            [[100, 100, 100], [50, 50, 50]],
            [[1.0, NAN, NAN], [NAN, 1.0, NAN]],
            in_universe=[[True, False, False], [True, True, True]],
        )
        eq(r.order_size[1], [-10, 20])
        eq(r.position[2], [0, 20])
        eq(r.cash, [1000, 0, 0])
        eq(r.equity, [1000, 1000, 1000])

    def test_reentry_after_a_gap(self):
        r = run(
            [[100, 100, 100, 100, 100]],
            [[1.0, NAN, NAN, 1.0, NAN]],
            in_universe=[[True, True, False, True, True]],
        )
        eq(r.order_size[:, A], [10, NAN, -10, 10, NAN])
        eq(r.position[:, A], [0, 10, 10, 0, 10])

    def test_close_order_on_the_last_date_never_fills(self):
        r = run([[100, 110, 120]], [[1.0, NAN, NAN]], in_universe=[[True, True, False]])
        eq(r.order_size[:, A], [10, NAN, -10])
        eq(r.position[:, A], [0, 10, 10])
        eq(r.equity, [1000, 1100, 1200])

    def test_position_held_to_the_end_in_universe_stays_open(self):
        r = run([[100, 110, 120]], [[1.0, NAN, NAN]])
        eq(r.order_size[:, A], [10, NAN, NAN])
        eq(r.position[:, A], [0, 10, 10])


class TestMissingPrices:
    def test_delisted_position_is_written_off_at_zero(self):
        r = run([[100, 110, NAN, NAN]], [[1.0, NAN, NAN, NAN]])
        eq(r.order_price[:, A], [100, NAN, 0, NAN])
        eq(r.order_size[:, A], [10, NAN, -10, NAN])
        eq(r.trade_price[:, A], [NAN, 100, NAN, 0])
        eq(r.trade_size[:, A], [NAN, 10, NAN, -10])
        eq(r.position[:, A], [0, 10, 10, 0])
        eq(r.cash, [1000, 0, 0, 0])
        # the loss lands on the delisting date, not on the close order's fill
        eq(r.equity, [1000, 1100, 0, 0])

    def test_committed_order_fills_into_a_delisting(self):
        # the order placed at 100 is honoured although the product is gone by
        # the fill date: buying it and losing it is the no-look-ahead outcome
        r = run([[100, NAN, NAN]], [[1.0, NAN, NAN]])
        eq(r.trade_price[:, A], [NAN, 100, 0])
        eq(r.trade_size[:, A], [NAN, 10, -10])
        eq(r.position[:, A], [0, 10, 0])
        eq(r.equity, [1000, 0, 0])

    def test_delisting_only_costs_that_products_value(self):
        r = run(
            [[100, 100, NAN, NAN], [50, 50, 60, 60]],
            [[0.5, NAN, NAN, NAN], [0.5, NAN, NAN, NAN]],
        )
        eq(r.position[:, B], [0, 10, 10, 10])
        eq(r.order_size[:, B], [10, NAN, NAN, NAN])
        eq(r.equity, [1000, 1000, 600, 600])

    def test_product_listed_mid_period(self):
        r = run([[100, 100, 100], [NAN, 50, 50]], [[0.5, NAN, NAN], [NAN, 0.5, NAN]])
        eq(r.order_size[:, B], [NAN, 5, NAN])
        eq(r.position[:, B], [0, 0, 5])
        eq(r.avg_price[:, B], [NAN, NAN, 50])
        eq(r.equity, [1000, 1000, 1000])


class Case(NamedTuple):
    price: FloatArray  # [T, P]
    weight: FloatArray  # [T, P]
    in_universe: BoolArray  # [T, P]
    result: Result


def _random_case(seed: int, T: int = 60, P: int = 4) -> Case:
    """A random long-only panel shaped like the mixin's: ragged lifetimes,
    universe gaps, and weights only where the product is in the universe."""
    rng = np.random.default_rng(seed)
    price = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.03, (T, P)), axis=0))
    for j in range(1, P):  # product 0 always trades, so the panel is never empty
        if rng.random() < 0.3:  # listed mid-period
            price[: int(rng.integers(1, 10)), j] = NAN
        if rng.random() < 0.3:  # delisted mid-period
            price[T - int(rng.integers(1, 20)) :, j] = NAN
    in_universe = ~np.isnan(price)
    for j in range(P):
        if rng.random() < 0.5:
            start = int(rng.integers(1, T - 10))
            in_universe[start : start + int(rng.integers(1, 10)), j] = False
    weight = np.full((T, P), NAN)
    for t in range(T):
        if rng.random() < 0.3:
            w = rng.random(P)
            w[rng.random(P) < 0.2] = 0.0  # some closes
            if w.sum() > 0.0:
                # gross exposure stays below 100%, so cash and therefore free
                # capital stay positive and the kernel never raises
                w *= rng.uniform(0.3, 0.9) / w.sum()
            w[rng.random(P) < 0.3] = NAN  # some holds
            weight[t] = np.where(in_universe[t], w, NAN)
    result = Result(
        *portfolio_backtest_loop_kernel(price, weight, in_universe, CAPITAL, True, T, P)
    )
    return Case(price, weight, in_universe, result)


@pytest.mark.parametrize("seed", range(100))
class TestInvariants:
    """Properties that must hold for any input, checked on random panels."""

    def test_position_is_the_running_sum_of_trades(self, seed: int):
        r = _random_case(seed).result
        np.testing.assert_allclose(
            r.position, np.cumsum(np.nan_to_num(r.trade_size), axis=0), atol=1e-9
        )

    def test_trade_price_and_size_come_together(self, seed: int):
        r = _random_case(seed).result
        np.testing.assert_array_equal(np.isnan(r.trade_price), np.isnan(r.trade_size))
        np.testing.assert_array_equal(np.isnan(r.order_price), np.isnan(r.order_size))
        assert np.all(np.isnan(r.trade_size[0]))
        assert not np.any(r.trade_size == 0.0)
        assert not np.any(r.order_size == 0.0)

    def test_every_order_fills_in_full_on_the_next_date(self, seed: int):
        r = _random_case(seed).result
        np.testing.assert_array_equal(r.trade_size[1:], r.order_size[:-1])
        np.testing.assert_array_equal(r.trade_price[1:], r.order_price[:-1])

    def test_avg_price_exists_exactly_while_in_position(self, seed: int):
        r = _random_case(seed).result
        np.testing.assert_array_equal(np.isnan(r.avg_price), r.position == 0.0)
        assert np.all(r.avg_price[r.position != 0.0] > 0.0)

    def test_cash_is_initial_capital_minus_the_cost_of_all_trades(self, seed: int):
        r = _random_case(seed).result
        spent = np.nan_to_num(r.trade_size * r.trade_price).sum(axis=1)
        np.testing.assert_allclose(r.cash, CAPITAL - np.cumsum(spent), atol=1e-6)

    def test_equity_is_cash_plus_positions_at_the_close(self, seed: int):
        case = _random_case(seed)
        r = case.result
        # an unpriced (delisted) position is worth nothing
        value = (r.position * np.nan_to_num(case.price)).sum(axis=1)
        np.testing.assert_allclose(r.equity, r.cash + value, atol=1e-6)

    def test_orders_are_priced_at_the_close_or_zero_if_delisted(self, seed: int):
        case = _random_case(seed)
        r = case.result
        placed = ~np.isnan(r.order_price)
        np.testing.assert_array_equal(
            r.order_price[placed], np.nan_to_num(case.price)[placed]
        )

    def test_nothing_is_ordered_without_an_instruction(self, seed: int):
        case = _random_case(seed)
        hold = case.in_universe & np.isnan(case.weight)
        assert np.all(np.isnan(case.result.order_size[hold]))

    def test_position_outside_the_universe_is_closed_on_the_next_date(self, seed: int):
        case = _random_case(seed)
        r = case.result
        out = ~case.in_universe
        held = out & (r.position != 0.0)
        np.testing.assert_array_equal(r.order_size[held], -r.position[held])
        assert np.all(np.isnan(r.order_size[out & (r.position == 0.0)]))
        assert np.all(r.position[1:][out[:-1]] == 0.0)

    def test_weights_land_on_target_fractions_of_free_capital(self, seed: int):
        case = _random_case(seed)
        r = case.result
        hold = case.in_universe & np.isnan(case.weight)
        drift = np.where(hold, r.position * np.nan_to_num(case.price), 0.0).sum(axis=1)
        free = r.equity - drift
        explicit = case.in_universe & ~np.isnan(case.weight)
        for t, j in np.argwhere(explicit[:-1]):
            target = case.weight[t, j] * free[t] / case.price[t, j]
            np.testing.assert_allclose(r.position[t + 1, j], target, atol=1e-9)

    def test_product_order_does_not_change_the_result(self, seed: int):
        case = _random_case(seed)
        T, P = case.price.shape
        perm = np.random.default_rng(seed).permutation(P)
        shuffled = Result(
            *portfolio_backtest_loop_kernel(
                np.ascontiguousarray(case.price[:, perm]),
                np.ascontiguousarray(case.weight[:, perm]),
                np.ascontiguousarray(case.in_universe[:, perm]),
                CAPITAL,
                True,
                T,
                P,
            )
        )
        for out, out_shuffled in zip(case.result[:6], shuffled[:6], strict=True):
            np.testing.assert_allclose(
                out[:, perm], out_shuffled, rtol=1e-9, atol=1e-9, equal_nan=True
            )
        np.testing.assert_allclose(case.result.equity, shuffled.equity, rtol=1e-9)
