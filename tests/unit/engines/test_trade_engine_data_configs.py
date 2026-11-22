"""Tests how the trade engine collects data configs across components before setting up the feeds."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

import pytest
from pfund_blocks.datas import BarData
from pfund_blocks.entities.products.product_base import BaseProduct
from pfund_blocks.resolution import Resolution

import pfund as pf
from pfund.data_config import DataConfig
from pfund.engines.settings.trade_engine_settings import TradeEngineSettings
from pfund.engines.trade_engine import TradeEngine


class MyStrategy(pf.Strategy):
    def trade(self, X: Any) -> list[Any]:
        return []


def create_data(source: str = "BYBIT") -> BarData:
    product = BaseProduct.model_validate(
        {"source": "BYBIT", "basis": "BTC_USDT_PERP", "symbol": "BTCUSDT"}
    )
    return BarData(product, Resolution("1m"), source=source)


@pytest.fixture
def engine() -> TradeEngine:
    # NOTE: PAPER, not the default SANDBOX, ENGINE_SETTINGS has no SANDBOX entry yet
    return TradeEngine(env="PAPER", settings=TradeEngineSettings())


def add_strategy_with(
    engine: TradeEngine, mocker: MockerFixture, name: str, data_configs: dict[Any, DataConfig]
) -> Any:
    strategy = engine.add_strategy(MyStrategy(), "1m", name=name)
    mocker.patch.object(strategy, "get_data_configs", return_value=data_configs)
    return strategy


def test_same_data_with_same_stream_settings_is_one_feed(
    engine: TradeEngine, mocker: MockerFixture
):
    add_strategy_with(engine, mocker, "a", {create_data(): DataConfig(num_stream_workers=2)})
    add_strategy_with(engine, mocker, "b", {create_data(): DataConfig(num_stream_workers=2)})

    data_configs = engine._get_all_data_configs()

    assert list(data_configs) == [create_data()]
    assert data_configs[create_data()].num_stream_workers == 2


def test_same_data_with_different_stream_settings_is_rejected(
    engine: TradeEngine, mocker: MockerFixture
):
    add_strategy_with(engine, mocker, "a", {create_data(): DataConfig(num_stream_workers=1)})
    add_strategy_with(engine, mocker, "b", {create_data(): DataConfig(num_stream_workers=2)})

    with pytest.raises(ValueError, match="'a' and 'b' with different stream settings"):
        engine._get_all_data_configs()


def test_same_product_from_different_sources_are_different_feeds(
    engine: TradeEngine, mocker: MockerFixture
):
    add_strategy_with(engine, mocker, "a", {create_data("BYBIT"): DataConfig(num_stream_workers=1)})
    add_strategy_with(
        engine, mocker, "b", {create_data("DATABENTO"): DataConfig(num_stream_workers=2)}
    )

    data_configs = engine._get_all_data_configs()

    assert {cast("BarData", data).source for data in data_configs} == {"BYBIT", "DATABENTO"}
    assert engine._get_all_datas() == set(data_configs)
