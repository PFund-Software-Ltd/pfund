"""Tests the add-data flow under the backtest engine: component.add_data() -> _add_product() -> MarketDataStore.add_data().

In backtesting, the venue only streams the component's resolution (see BacktestMixin._get_streaming_resolutions),
so every other bar resolution is auto-resampled from it.
"""

from __future__ import annotations

from typing import Any

import pytest
from pfund_blocks.datas import BarData, TickData
from pfund_blocks.resolution import Resolution

import pfund as pf
from pfund.data_config import DataConfig
from pfund.engines.backtest_engine import BacktestEngine
from pfund.engines.settings.backtest_engine_settings import BacktestEngineSettings
from pfund.simulated import SimulatedVenue

pytest.importorskip("pfund_bybit", reason="the add-data flow needs an installed venue")

VENUE = "BYBIT"
BTC = "BTC_USDT_PERP"
ETH = "ETH_USDT_PERP"
BTC_NAME = "BYBIT_BTC_USDT_PERPETUAL"


class MyStrategy(pf.Strategy):
    def trade(self, X: Any) -> list[Any]:
        return []


@pytest.fixture
def strategy() -> Any:
    """A backtest strategy at 1m, adopted by the engine, so add_data() runs the backtest flow."""
    engine = BacktestEngine(data_range="1y", settings=BacktestEngineSettings())
    return engine.add_strategy(MyStrategy(), "1m", name="alpha")


class TestProduct:
    def test_product_is_created_by_the_simulated_venue(self, strategy: Any):
        assert strategy._get_venue_class(VENUE) is SimulatedVenue.for_venue(VENUE)

        strategy.add_data(VENUE, BTC)

        product = strategy.products[BTC_NAME]
        assert product.source == VENUE
        assert str(product.basis).startswith("BTC_USDT")

    def test_custom_product_name_is_used(self, strategy: Any):
        datas = strategy.add_data(VENUE, BTC, product_name="btc")

        assert list(strategy.products) == ["btc"]
        assert datas[0].product.name == "btc"
        assert strategy.get_data("btc", "1m") is datas[0]

    def test_same_name_for_a_different_product_is_rejected(self, strategy: Any):
        strategy.add_data(VENUE, BTC, product_name="coin")

        with pytest.raises(AssertionError, match="already used"):
            strategy.add_data(VENUE, ETH, product_name="coin")


class TestResolutions:
    def test_component_resolution_is_always_added(self, strategy: Any):
        datas = strategy.add_data(VENUE, BTC)

        assert [data.resolution for data in datas] == [Resolution("1m")]
        assert isinstance(datas[0], BarData)

    def test_unstreamed_resolution_is_auto_resampled_from_the_component_resolution(
        self, strategy: Any
    ):
        datas = strategy.add_data(VENUE, BTC, resolutions=["6m"])

        data_1m, data_6m = datas
        assert data_6m.get_resampler() is data_1m
        assert data_1m.get_resamplees() == [data_6m]
        config = strategy.market_data_store.get_config(data_6m)
        assert config.resample == {Resolution("6m"): Resolution("1m")}

    def test_shift_is_applied_to_the_bar_data(self, strategy: Any):
        strategy.add_data(
            VENUE, BTC, resolutions=["1h"], config=DataConfig(shift={"1h": "30m"})
        )

        data_1h = strategy.get_data(BTC_NAME, "1h")
        assert isinstance(data_1h, BarData)
        assert data_1h.shift == "30m"
        # a shifted resolution is always resampled, the venue's bars are not shifted
        assert data_1h.get_resampler() is strategy.get_data(BTC_NAME, "1m")

    def test_tick_resolution_creates_tick_data(self, strategy: Any):
        strategy.add_data(
            VENUE, BTC, resolutions=["1t"], config=DataConfig(auto_resample=False)
        )

        assert isinstance(strategy.get_data(BTC_NAME, "1t"), TickData)

    def test_weekly_resolution_is_rejected(self, strategy: Any):
        with pytest.raises(ValueError, match="at least daily"):
            strategy.add_data(VENUE, BTC, resolutions=["1w"])


class TestOneCallPerProduct:
    def test_adding_the_same_product_twice_is_rejected(self, strategy: Any):
        strategy.add_data(VENUE, BTC, resolutions=["6m"])

        with pytest.raises(ValueError, match="has already been added"):
            strategy.add_data(VENUE, BTC, resolutions=["1h"])
        assert strategy.get_data(BTC_NAME, "1h") is None

    def test_different_products_can_be_added(self, strategy: Any):
        strategy.add_data(VENUE, BTC)
        strategy.add_data(VENUE, ETH)

        assert len(strategy.products) == 2
        assert len(strategy.market_data_store.get_datas()) == 2

    def test_all_data_of_a_product_share_one_resolved_config(self, strategy: Any):
        config = DataConfig()
        datas = strategy.add_data(VENUE, BTC, resolutions=["6m", "1h"], config=config)

        store = strategy.market_data_store
        resolved = store.get_config(datas[0])
        assert all(store.get_config(data) is resolved for data in datas)
        # the user's config is not mutated by auto-resampling
        assert config.resample == {}


class TestAsFeatures:
    def test_as_features_defaults_to_true(self, strategy: Any):
        strategy.add_data(VENUE, BTC)

        assert strategy.market_data_store.data_as_features is True

    def test_as_features_false_is_kept(self, strategy: Any):
        strategy.add_data(VENUE, BTC, as_features=False)

        assert strategy.market_data_store.data_as_features is False

    def test_inconsistent_as_features_is_rejected(self, strategy: Any):
        strategy.add_data(VENUE, BTC, as_features=True)

        with pytest.raises(ValueError, match="data_as_features"):
            strategy.add_data(VENUE, ETH, as_features=False)
