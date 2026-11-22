from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from pathlib import Path
    from unittest.mock import MagicMock

    from pfeed.dataflow.dataflow import DataFlow
    from pytest_mock import MockerFixture

import datetime
from dataclasses import dataclass, field
from types import SimpleNamespace

import polars as pl
import pytest
from pfeed.dataflow.result import DataFlowResult, RunResult
from pfeed.enums import DataAccessType, DataLayer
from pfeed.errors import DataNotFoundError, MissingDatesError
from pfeed.io.parquet_io import ParquetIO
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


# --- materialize() ---


DATA_START = datetime.date(2024, 1, 1)
DATA_END = datetime.date(2024, 1, 3)


def create_bars(product: BaseProduct, days: list[int] | None = None) -> pl.LazyFrame:
    """One 1m bar per day, keyed like the spine materialize() builds."""
    days = days or [1, 2, 3]
    return pl.LazyFrame(
        {
            "date": [datetime.datetime(2024, 1, day) for day in days],
            "product": [product.name] * len(days),
            "resolution": ["1_MINUTE"] * len(days),
            "close": [float(day) for day in days],
        }
    )


@dataclass
class Stored:
    """What a fake storage holds for one product."""

    df: pl.LazyFrame | None
    missing_dates: list[datetime.date] = field(default_factory=list)
    # a per-date dataflow failed while reading, e.g. a corrupt file
    failed: bool = False


class FakeMarketFeed:
    """Stands in for pfeed's pipeline-mode MarketFeed.

    retrieve()/download() queue a request and return the feed, run() returns its RunResult,
    like pfeed does in pipeline mode. Storages are keyed by the product basis passed to retrieve().
    """

    def __init__(self, cache_io: ParquetIO):
        self.cache_io = cache_io
        self.cache: dict[str, Stored] = {}
        self.storage: dict[str, Stored] = {}
        self.source: dict[str, pl.LazyFrame | None] = {}
        # whether the original storage only has finer data, i.e. retrieve() resamples
        self.resampled = False
        self.calls: list[tuple[str, str]] = []  # (verb, "cache" | "storage" | "source")
        self.cache_writes: list[
            str
        ] = []  # products whose retrieved data is loaded into the cache
        self._pending: tuple[str, Stored | None] | None = None
        self.data_source: SimpleNamespace | None = None

    def set_num_workers(self, num_workers: int | None) -> None:
        pass

    def retrieve(
        self,
        *,
        product: str,
        io: ParquetIO,
        data_layer: DataLayer = DataLayer.CLEANED,
        raise_on_missing_data: bool = False,
        **kwargs: Any,
    ) -> FakeMarketFeed:
        is_cache = io is self.cache_io
        assert data_layer == (DataLayer.CURATED if is_cache else DataLayer.CLEANED)
        storage = self.cache if is_cache else self.storage
        stored = storage.get(product)
        # like pfeed's probing: nothing stored means every date is missing
        missing_dates = stored.missing_dates if stored else [DATA_START, DATA_END]
        if missing_dates and raise_on_missing_data:
            raise MissingDatesError(f"{product} is not fully stored", missing_dates)
        self.calls.append(("retrieve", "cache" if is_cache else "storage"))
        self._pending = (product, stored)
        return self

    def load(
        self, io: ParquetIO | None = None, data_layer: DataLayer | None = None
    ) -> FakeMarketFeed:
        assert self._pending is not None
        if io is not None:
            assert io is self.cache_io and data_layer == DataLayer.CURATED
            self.cache_writes.append(self._pending[0])
        return self

    def download(self, *, product: str, **kwargs: Any) -> FakeMarketFeed:
        self.calls.append(("download", "source"))
        df = self.source.get(product)
        self._pending = (product, Stored(df) if df is not None else None)
        return self

    def run(self) -> RunResult:
        assert self._pending is not None, "run() without a queued request"
        _, stored = self._pending
        self._pending = None
        if stored is None:
            return RunResult(data=None, dataflows=[])
        dataflows = (
            [SimpleNamespace(result=DataFlowResult.failed(RuntimeError("read failed")))]
            if stored.failed
            else []
        )
        return RunResult(data=stored.df, dataflows=cast("list[DataFlow]", dataflows))

    def _get_current_request(self) -> SimpleNamespace:
        return SimpleNamespace(
            data_resolution=Resolution("1t") if self.resampled else Resolution("1m"),
            target_resolution=Resolution("1m"),
        )


@dataclass
class MaterializeEnv:
    store: MarketDataStore
    feed: FakeMarketFeed
    settings: SimpleNamespace
    data_source: SimpleNamespace
    storage_io: ParquetIO

    def add_product(self, product: BaseProduct) -> None:
        self.store.add_data(product, config=DataConfig(io=self.storage_io))
        self.store._component.products[product.name] = product


@pytest.fixture
def env(
    store: MarketDataStore, mocker: MockerFixture, tmp_path: Path
) -> MaterializeEnv:
    """A store with a fake pfeed source; by default caching is 'auto' and auto-download is on."""
    cache_io = ParquetIO(base_path=str(tmp_path / "cache"))
    feed = FakeMarketFeed(cache_io)
    data_source = SimpleNamespace(
        name="BYBIT",
        market_feed=feed,
        metadata=SimpleNamespace(access_type=DataAccessType.FREE),
    )
    feed.data_source = data_source
    mocker.patch("pfeed.registry.get_source", return_value=lambda **_: data_source)
    mocker.patch.object(store, "_create_io_for_caching", return_value=cache_io)
    settings = SimpleNamespace(cache_materialized_data="auto", auto_download_data=True)
    component = cast("MagicMock", store._component)
    component.products = {}
    component.context.settings = settings
    component.context.data_start = DATA_START
    component.context.data_end = DATA_END
    return MaterializeEnv(
        store=store,
        feed=feed,
        settings=settings,
        data_source=data_source,
        storage_io=ParquetIO(base_path=str(tmp_path / "storage")),
    )


def get_closes(store: MarketDataStore) -> list[float]:
    return cast("pl.DataFrame", store.get_df(to_native=True))["close"].to_list()


class TestMaterialize:
    def test_no_data_sets_no_df(self, env: MaterializeEnv) -> None:
        env.store.materialize()

        assert env.store._df is None
        assert env.feed.calls == []

    def test_complete_cache_skips_original_storage(self, env: MaterializeEnv) -> None:
        product = create_product()
        env.add_product(product)
        env.feed.cache[str(product.basis)] = Stored(create_bars(product))
        env.feed.storage[str(product.basis)] = Stored(create_bars(product, [9]))

        env.store.materialize()

        assert env.feed.calls == [("retrieve", "cache")]
        assert get_closes(env.store) == [1.0, 2.0, 3.0]

    def test_partial_cache_falls_through_to_original_storage(
        self, env: MaterializeEnv
    ) -> None:
        product = create_product()
        env.add_product(product)
        # the cache only has day 2, the original storage has the full range
        env.feed.cache[str(product.basis)] = Stored(
            create_bars(product, [2]), missing_dates=[DATA_START, DATA_END]
        )
        env.feed.storage[str(product.basis)] = Stored(create_bars(product))

        env.store.materialize()

        assert env.feed.calls == [("retrieve", "storage")]
        assert get_closes(env.store) == [1.0, 2.0, 3.0]

    def test_failed_cache_read_falls_through_to_original_storage(
        self, env: MaterializeEnv
    ) -> None:
        product = create_product()
        env.add_product(product)
        env.feed.cache[str(product.basis)] = Stored(
            create_bars(product, [2]), failed=True
        )
        env.feed.storage[str(product.basis)] = Stored(create_bars(product))

        env.store.materialize()

        assert env.feed.calls == [("retrieve", "cache"), ("retrieve", "storage")]
        assert get_closes(env.store) == [1.0, 2.0, 3.0]

    def test_disabled_cache_is_not_read(self, env: MaterializeEnv) -> None:
        env.settings.cache_materialized_data = False
        product = create_product()
        env.add_product(product)
        env.feed.cache[str(product.basis)] = Stored(create_bars(product, [9]))
        env.feed.storage[str(product.basis)] = Stored(create_bars(product))

        env.store.materialize()

        assert env.feed.calls == [("retrieve", "storage")]
        assert get_closes(env.store) == [1.0, 2.0, 3.0]

    @pytest.mark.parametrize(
        ("cache_materialized_data", "resampled", "is_cached"),
        [
            ("auto", True, True),
            ("auto", False, False),
            (True, False, True),
            (False, True, False),
        ],
    )
    def test_caches_retrieved_data_per_setting(
        self,
        env: MaterializeEnv,
        cache_materialized_data: bool | str,
        resampled: bool,
        is_cached: bool,
    ) -> None:
        env.settings.cache_materialized_data = cache_materialized_data
        env.feed.resampled = resampled
        product = create_product()
        env.add_product(product)
        env.feed.storage[str(product.basis)] = Stored(create_bars(product))

        env.store.materialize()

        assert env.feed.cache_writes == ([str(product.basis)] if is_cached else [])

    def test_missing_data_is_auto_downloaded(self, env: MaterializeEnv) -> None:
        product = create_product()
        env.add_product(product)
        env.feed.source[str(product.basis)] = create_bars(product)

        env.store.materialize()

        assert env.feed.calls[-1] == ("download", "source")
        assert get_closes(env.store) == [1.0, 2.0, 3.0]

    def test_missing_data_raises_when_auto_download_is_disabled(
        self, env: MaterializeEnv
    ) -> None:
        env.settings.auto_download_data = False
        env.add_product(create_product())

        with pytest.raises(DataNotFoundError, match="auto_download_data"):
            env.store.materialize()
        assert ("download", "source") not in env.feed.calls

    def test_paid_by_usage_data_is_not_auto_downloaded(
        self, env: MaterializeEnv
    ) -> None:
        env.data_source.metadata.access_type = DataAccessType.PAID_BY_USAGE
        env.add_product(create_product())

        with pytest.raises(DataNotFoundError, match="PAID"):
            env.store.materialize()
        assert ("download", "source") not in env.feed.calls

    def test_failed_download_raises(self, env: MaterializeEnv) -> None:
        env.add_product(create_product())

        with pytest.raises(DataNotFoundError, match="Failed to download"):
            env.store.materialize()

    def test_products_are_combined_into_one_sorted_spine(
        self, env: MaterializeEnv
    ) -> None:
        btc = create_product()
        eth = create_product(basis="ETH_USDT_PERP", symbol="ETHUSDT")
        env.add_product(eth)
        env.add_product(btc)
        env.feed.storage[str(btc.basis)] = Stored(create_bars(btc))
        env.feed.storage[str(eth.basis)] = Stored(create_bars(eth))

        env.store.materialize()

        df = cast("pl.DataFrame", env.store.get_df(to_native=True))
        assert df.columns[:3] == ["date", "product", "resolution"]
        assert df["source_type"].unique().to_list() == ["BATCH"]
        # sorted by (date, product, resolution)
        assert df.select("date", "product").rows() == sorted(
            df.select("date", "product").rows()
        )
        assert len(df) == 6
