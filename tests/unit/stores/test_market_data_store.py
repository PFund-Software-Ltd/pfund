from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

import pytest
from pfund_blocks.datas import BarData
from pfund_blocks.entities.products.product_base import BaseProduct
from pfund_blocks.resolution import Resolution

from pfund.data_config import DataConfig
from pfund.stores.market_data_store import MarketDataStore


def create_product(
    basis: str = "BTC_USDT_PERP", symbol: str = "BTCUSDT"
) -> BaseProduct:
    return BaseProduct.model_validate(
        {"source": "BYBIT", "basis": basis, "symbol": symbol}
    )


@pytest.fixture
def store(mocker: MockerFixture) -> MarketDataStore:
    """A store whose component runs at 1m and whose venue streams 1m and 3m bars."""
    databoy = mocker.MagicMock()
    databoy._component.resolution = Resolution("1m")
    databoy._component._get_streaming_resolutions.return_value = [
        Resolution("1m"),
        Resolution("3m"),
    ]
    return MarketDataStore(databoy)


def test_add_data_applies_shift_to_bar_data(store: MarketDataStore) -> None:
    product = create_product()
    store.add_data(product, resolutions=["1h"], config=DataConfig(shift={"1h": "30m"}))

    data = store.get_data(product.name, "1h")
    assert isinstance(data, BarData)
    assert data.shift == "30m"
    # an unshifted resolution stays unshifted
    data_1m = store.get_data(product.name, "1m")
    assert isinstance(data_1m, BarData)
    assert data_1m.shift == ""


def test_add_data_rejects_same_product_twice(store: MarketDataStore) -> None:
    product = create_product()
    store.add_data(product, resolutions=["6m"])

    with pytest.raises(ValueError, match="has already been added"):
        store.add_data(product, resolutions=["1h"])
    # the second call adds nothing
    assert store.get_data(product.name, "1h") is None
    assert store.get_data(product.name, "6m") is not None


def test_add_data_shares_one_resolved_config_per_product(
    store: MarketDataStore,
) -> None:
    product = create_product()
    config = DataConfig()
    datas = store.add_data(product, resolutions=["6m"], config=config)

    # 6m isn't streamed, so it's auto-resampled from the coarsest fitting streamed resolution
    resolved = store.get_config(datas[0])
    assert resolved.resample == {Resolution("6m"): Resolution("3m")}
    assert all(store.get_config(data) is resolved for data in store.get_datas())
    # the user's config is not mutated
    assert config.resample == {}


def test_add_data_allows_different_products(store: MarketDataStore) -> None:
    btc = create_product()
    eth = create_product(basis="ETH_USDT_PERP", symbol="ETHUSDT")
    store.add_data(btc)
    store.add_data(eth)

    assert store.get_data(btc.name, "1m") is not None
    assert store.get_data(eth.name, "1m") is not None


def test_add_data_rejects_inconsistent_as_features(store: MarketDataStore) -> None:
    store.add_data(create_product(), as_features=True)

    with pytest.raises(ValueError, match="data_as_features"):
        store.add_data(
            create_product(basis="ETH_USDT_PERP", symbol="ETHUSDT"), as_features=False
        )
    assert store.data_as_features is True
