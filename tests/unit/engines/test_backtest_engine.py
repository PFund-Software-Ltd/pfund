"""Tests BacktestEngine: construction, the add-component and add-data flows, build() and run(), plus the
strategy's chunked vectorized backtest, i.e. strategy.backtest(X) chunking the df (sequentially or over Ray).

engine.build() gathers and builds the components, the component side (_gather/_build/start/stop) is
stubbed in the build() and run() tests. In the strategy.backtest(X) tests, the per-chunk backtest (_backtest_chunk) is
stubbed, only the chunking is tested here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from pytest_mock import MockerFixture

import contextlib
import datetime

import polars as pl
import pytest
from pfund_blocks.datas import BarData, TickData
from pfund_blocks.enums.env import Environment
from pfund_blocks.resolution import Resolution
from sklearn.linear_model import LinearRegression

import pfund as pf
from pfund.backtest._mixins.backtest_mixin import BacktestMixin
from pfund.backtest._mixins.feature_mixin import BacktestFeatureMixin
from pfund.backtest._mixins.model_mixin import BacktestModelMixin
from pfund.backtest._mixins.strategy_mixin import (
    BacktestStrategyMixin,
    _split_df_into_time_chunks,
    _without_dfs,
)
from pfund.backtest.cv.dataset_splits import DatasetSplits
from pfund.components._mixins.component_mixin import ComponentMixin
from pfund.components._mixins.dataframe_mixin import DataFrameMixin
from pfund.components.actor_proxy import ActorProxy
from pfund.components.models.sklearn_model import SKLearnModel
from pfund.components.strategies._dummy_strategy import _DummyStrategy
from pfund.data_config import DataConfig
from pfund.engines.backtest_engine import BacktestEngine
from pfund.engines.settings.backtest_engine_settings import BacktestEngineSettings
from pfund.enums import BacktestMode, RunMode
from pfund.simulated import SimulatedVenue

VENUE = "BYBIT"
BTC = "BTC_USDT_PERP"
ETH = "ETH_USDT_PERP"
BTC_NAME = "BYBIT_BTC_USDT_PERPETUAL"
CHUNK_COL = BacktestStrategyMixin.CHUNK_COL


class MyStrategy(pf.Strategy):
    def trade(self, X: Any) -> list[Any]:
        return []


class MyFeature(pf.Feature):
    def transform(self, X: Any, *args: Any, **kwargs: Any) -> Any:
        return [0]


def _is_backtestee(component: Any) -> bool:
    return component._is_backtestee


@pytest.fixture
def engine() -> BacktestEngine:
    return BacktestEngine(data_range="1y", settings=BacktestEngineSettings())


@pytest.fixture
def strategy(engine: BacktestEngine) -> Any:
    """A backtest strategy at 1m, adopted by the engine, so add_data() runs the backtest flow."""
    return engine.add_strategy(MyStrategy(), "1m", name="alpha")


@pytest.fixture
def bybit() -> None:
    pytest.importorskip(
        "pfund_bybit", reason="the add-data flow needs an installed venue"
    )


@pytest.fixture
def create_actor(mocker: MockerFixture) -> Any:
    return mocker.patch.object(ActorProxy, "_create_actor")


class TestInit:
    def test_defaults(self, engine: BacktestEngine):
        assert engine.env == Environment.BACKTEST
        assert engine.backtest_mode == BacktestMode.VECTORIZED
        assert engine.context.dataset_splits is None
        assert not engine.is_running()

    def test_mode_accepts_a_string(self):
        engine = BacktestEngine(settings=BacktestEngineSettings(), mode="vectorized")

        assert engine.backtest_mode == BacktestMode.VECTORIZED

    def test_dataset_splits_is_resolved_into_splits(self):
        engine = BacktestEngine(settings=BacktestEngineSettings(), dataset_splits=721)

        assert isinstance(engine.context.dataset_splits, DatasetSplits)
        assert engine.context.dataset_splits.is_holdout


class TestEngineAddStrategy:
    def test_strategy_is_adopted_hydrated_and_top(self, engine: BacktestEngine):
        strategy = MyStrategy()

        added = engine.add_strategy(strategy, "1d", name="alpha")

        assert added is strategy
        assert isinstance(strategy, BacktestStrategyMixin)
        assert isinstance(strategy, MyStrategy)
        assert strategy.name == "alpha"
        assert strategy.run_mode == RunMode.LOCAL
        assert strategy.resolution == Resolution("1d")
        assert strategy.df_form == "long"
        assert _is_backtestee(strategy)
        assert engine.get_strategy("alpha") is strategy

    def test_ray_args_are_ignored_with_a_warning(
        self, engine: BacktestEngine, mocker: MockerFixture, create_actor: Any
    ):
        warning = mocker.patch.object(engine._logger, "warning")
        strategy = MyStrategy()

        added = engine.add_strategy(
            strategy, "1d", ray_actor_options={"name": "x"}, num_gpus=1
        )

        assert added is strategy
        assert strategy.run_mode == RunMode.LOCAL
        warning.assert_called_once()
        assert "ignored in backtesting" in warning.call_args.args[0]
        create_actor.assert_not_called()

    def test_only_one_strategy_is_allowed(self, engine: BacktestEngine):
        engine.add_strategy(MyStrategy(), "1d")

        with pytest.raises(Exception, match="only one strategy is allowed"):
            engine.add_strategy(MyStrategy(), "1d", name="other")

    def test_dummy_strategy_name_is_reserved(self, engine: BacktestEngine):
        with pytest.raises(ValueError, match="is reserved"):
            engine.add_strategy(MyStrategy(), "1d", name=_DummyStrategy.__name__)

    def test_strategy_after_top_level_model_is_rejected(self, engine: BacktestEngine):
        engine.add_model(LinearRegression(), "1d")

        with pytest.raises(Exception, match="is already being backtested"):
            engine.add_strategy(MyStrategy(), "1d")


class TestEngineAddTopLevelModelFeature:
    def test_model_is_hosted_by_dummy_strategy(self, engine: BacktestEngine):
        model = engine.add_model(LinearRegression(), "1d", name="lr")

        dummy = engine.get_strategy(_DummyStrategy.__name__)
        assert isinstance(dummy, _DummyStrategy)
        assert isinstance(dummy, BacktestStrategyMixin)
        assert dummy.resolution == Resolution("1d")
        assert isinstance(model, SKLearnModel)
        assert isinstance(model, BacktestModelMixin)
        assert model.name == "lr"
        assert model.df_form == "wide"
        assert dummy.models == {"lr": model}
        assert _is_backtestee(model)

    def test_feature_is_hosted_by_dummy_strategy(self, engine: BacktestEngine):
        feature = MyFeature()

        added = engine.add_feature(feature, "1d", name="ema", df_form="long")

        dummy = engine.get_strategy(_DummyStrategy.__name__)
        assert added is feature
        assert isinstance(feature, BacktestFeatureMixin)
        assert feature.name == "ema"
        assert feature.df_form == "long"
        assert dummy.features == {"ema": feature}
        assert _is_backtestee(feature)

    @pytest.mark.parametrize("second", ["model", "feature"])
    def test_only_one_top_level_model_or_feature_is_allowed(
        self, engine: BacktestEngine, second: str
    ):
        engine.add_feature(MyFeature(), "1d")

        with pytest.raises(ValueError, match="only one top-level model/feature"):
            if second == "model":
                engine.add_model(LinearRegression(), "1d")
            else:
                engine.add_feature(MyFeature(), "1d", name="other")

    def test_top_level_model_after_strategy_is_rejected(self, engine: BacktestEngine):
        engine.add_strategy(MyStrategy(), "1d")

        with pytest.raises(ValueError, match="cannot add a top-level model/feature"):
            engine.add_model(LinearRegression(), "1d")


class TestComponentAddChildren:
    @pytest.fixture
    def strategy(self, engine: BacktestEngine) -> MyStrategy:
        return engine.add_strategy(MyStrategy(), "1d")

    def test_child_feature_is_adopted_and_not_top(self, strategy: MyStrategy):
        feature = MyFeature()

        added = strategy.add_feature(feature, name="ema")

        assert added is feature
        assert isinstance(feature, BacktestFeatureMixin)
        assert isinstance(feature, MyFeature)
        assert feature.name == "ema"
        assert feature.resolution == strategy.resolution
        assert strategy.features == {"ema": feature}
        assert not _is_backtestee(feature)

    def test_child_model_is_wrapped_and_adopted(self, strategy: MyStrategy):
        model = strategy.add_model(LinearRegression())

        assert isinstance(model, SKLearnModel)
        assert isinstance(model, BacktestModelMixin)
        assert strategy.models == {"LinearRegression": model}
        assert not _is_backtestee(model)

    def test_substrategy_is_adopted_and_marked(self, strategy: MyStrategy):
        sub = MyStrategy()

        added = strategy.add_strategy(sub, name="sub")

        assert added is sub
        assert isinstance(sub, BacktestStrategyMixin)
        assert sub.is_substrategy()
        assert strategy.strategies == {"sub": sub}
        assert not _is_backtestee(sub)

    def test_grandchild_is_adopted(self, strategy: MyStrategy):
        parent = strategy.add_feature(MyFeature(), name="parent")
        assert parent is not None

        child = parent.add_feature(MyFeature(), name="child")

        assert isinstance(child, BacktestFeatureMixin)
        assert parent.features == {"child": child}

    def test_child_ray_args_are_ignored_with_a_warning(
        self, strategy: MyStrategy, mocker: MockerFixture, create_actor: Any
    ):
        warning = mocker.patch.object(strategy.logger, "warning")
        feature = MyFeature()

        added = strategy.add_feature(feature, name="ema", num_cpus=1)

        assert added is feature
        assert not isinstance(added, ActorProxy)
        assert feature.run_mode == RunMode.LOCAL
        warning.assert_called_once()
        assert "ignored in backtesting" in warning.call_args.args[0]
        create_actor.assert_not_called()

    @pytest.mark.parametrize("second_name", ["ema", "EMA"])
    def test_duplicate_child_name_is_rejected(
        self, strategy: MyStrategy, second_name: str
    ):
        strategy.add_feature(MyFeature(), name="ema")

        with pytest.raises(ValueError, match="already exists"):
            strategy.add_feature(MyFeature(), name=second_name)

    def test_finer_child_resolution_is_rejected(self, strategy: MyStrategy):
        with pytest.raises(ValueError, match="is finer than parent"):
            strategy.add_feature(MyFeature(), "1m")


@pytest.mark.usefixtures("bybit")
class TestAddDataProduct:
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


@pytest.mark.usefixtures("bybit")
class TestAddDataResolutions:
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


@pytest.mark.usefixtures("bybit")
class TestAddDataOneCallPerProduct:
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


@pytest.mark.usefixtures("bybit")
class TestAddDataAsFeatures:
    def test_as_features_defaults_to_true(self, strategy: Any):
        strategy.add_data(VENUE, BTC)

        assert strategy.market_data_store.data_as_features is True

    def test_as_features_false_is_overridden_in_vectorized_backtesting(
        self, strategy: Any, mocker: MockerFixture
    ):
        warning = mocker.patch.object(strategy.logger, "warning")

        strategy.add_data(VENUE, BTC, as_features=False)

        # the vectorized backtest needs OHLC in X
        assert strategy.market_data_store.data_as_features is True
        warning.assert_called_once()
        assert "as_features set to True" in warning.call_args.args[0]

    def test_as_features_false_is_kept_in_event_driven_backtesting(
        self, engine: BacktestEngine, strategy: Any, mocker: MockerFixture
    ):
        mocker.patch.object(engine.context, "mode", BacktestMode.EVENT_DRIVEN)

        strategy.add_data(VENUE, BTC, as_features=False)

        assert strategy.market_data_store.data_as_features is False

    def test_inconsistent_as_features_is_rejected(
        self, engine: BacktestEngine, strategy: Any, mocker: MockerFixture
    ):
        mocker.patch.object(engine.context, "mode", BacktestMode.EVENT_DRIVEN)
        strategy.add_data(VENUE, BTC, as_features=True)

        with pytest.raises(ValueError, match="data_as_features"):
            strategy.add_data(VENUE, ETH, as_features=False)


@pytest.mark.usefixtures("bybit")
class TestAddDataSourceAndOrigin:
    def test_source_defaults_to_the_product_venue(self, strategy: Any):
        datas = strategy.add_data(VENUE, BTC, resolutions=["6m"])

        assert all(data.source == VENUE for data in datas)
        assert all(data.origin == "" for data in datas)

    def test_data_source_overrides_the_venue(self, strategy: Any):
        datas = strategy.add_data(
            VENUE,
            BTC,
            resolutions=["6m"],
            config=DataConfig(data_source="databento", data_origin="glbx"),
        )

        # resamplers and resamplees come from the same source, so they can be bound
        assert all(data.source == "DATABENTO" for data in datas)
        assert all(data.origin == "glbx" for data in datas)
        assert datas[1].get_resampler() is datas[0]


def _dates(n: int) -> list[datetime.datetime]:
    start = datetime.datetime(2025, 1, 1)
    return [start + datetime.timedelta(minutes=i) for i in range(n)]


def _wide_df(num_dates: int) -> pl.DataFrame:
    """One row per date."""
    return pl.DataFrame({"date": _dates(num_dates), "x": list(range(num_dates))})


def _long_df(num_dates: int, products: list[str]) -> pl.DataFrame:
    """One row per (date, product), sorted by date first like KEY_COLS."""
    rows = [
        {"date": date, "product": product, "x": i}
        for i, date in enumerate(_dates(num_dates))
        for product in products
    ]
    return pl.DataFrame(rows)


def _chunk_dates(chunk: pl.DataFrame) -> list[datetime.datetime]:
    return chunk.get_column("date").unique(maintain_order=True).to_list()


class TestSplitDfIntoTimeChunks:
    def test_single_chunk_is_the_whole_df(self):
        df = _wide_df(5)
        chunks = _split_df_into_time_chunks(df, 1)
        assert len(chunks) == 1
        chunk, chunk_num = chunks[0]
        assert chunk_num == 0
        assert chunk.equals(df)

    def test_chunk_nums_are_in_order(self):
        chunks = _split_df_into_time_chunks(_wide_df(10), 4)
        assert [chunk_num for _, chunk_num in chunks] == [0, 1, 2, 3]

    def test_chunks_are_contiguous_and_cover_every_row(self):
        df = _wide_df(10)
        chunks = _split_df_into_time_chunks(df, 3)
        recombined = pl.concat([chunk for chunk, _ in chunks], how="vertical")
        assert recombined.equals(df)

    def test_chunk_sizes_follow_array_split(self):
        # 10 dates into 3 chunks -> 4, 3, 3 dates (np.array_split puts the extra dates first)
        chunks = _split_df_into_time_chunks(_wide_df(10), 3)
        assert [len(chunk) for chunk, _ in chunks] == [4, 3, 3]

    def test_long_df_never_splits_a_timestamp(self):
        products = ["BTC", "ETH", "SOL"]
        df = _long_df(7, products)
        chunks = _split_df_into_time_chunks(df, 3)

        seen: set[datetime.datetime] = set()
        for chunk, _ in chunks:
            dates = _chunk_dates(chunk)
            # no date appears in more than one chunk
            assert seen.isdisjoint(dates)
            seen.update(dates)
            # every date in a chunk has all products, i.e. the timestamp was not cut
            assert len(chunk) == len(dates) * len(products)
        assert len(seen) == 7

    def test_chunks_split_by_unique_dates_not_rows(self):
        # 4 dates x 2 products = 8 rows, 2 chunks -> 2 dates (4 rows) each
        chunks = _split_df_into_time_chunks(_long_df(4, ["BTC", "ETH"]), 2)
        assert [len(_chunk_dates(chunk)) for chunk, _ in chunks] == [2, 2]
        assert [len(chunk) for chunk, _ in chunks] == [4, 4]

    def test_num_chunks_equal_to_num_dates(self):
        chunks = _split_df_into_time_chunks(_long_df(3, ["BTC", "ETH"]), 3)
        assert [len(_chunk_dates(chunk)) for chunk, _ in chunks] == [1, 1, 1]

    def test_raises_when_num_chunks_exceeds_num_dates(self):
        # 6 rows but only 3 unique dates
        with pytest.raises(ValueError, match="cannot exceed the number of unique"):
            _split_df_into_time_chunks(_long_df(3, ["BTC", "ETH"]), 4)

    def test_raises_when_num_chunks_below_one(self):
        with pytest.raises(ValueError, match="greater than 0"):
            _split_df_into_time_chunks(_wide_df(3), 0)

    def test_raises_without_date_column(self):
        df = pl.DataFrame({"x": [1, 2, 3]})
        with pytest.raises(ValueError, match="requires a 'date' column"):
            _split_df_into_time_chunks(df, 2)

    def test_raises_on_empty_df(self):
        df = pl.DataFrame({"date": []}, schema={"date": pl.Datetime("us")})
        with pytest.raises(ValueError, match="at least one row"):
            _split_df_into_time_chunks(df, 1)

    def test_raises_on_null_dates(self):
        dates: list[datetime.datetime | None] = [*_dates(2), None]
        df = pl.DataFrame({"date": dates})
        with pytest.raises(ValueError, match="null dates"):
            _split_df_into_time_chunks(df, 2)

    def test_raises_when_not_sorted_by_date(self):
        df = pl.DataFrame({"date": list(reversed(_dates(3)))})
        with pytest.raises(ValueError, match="sorted by 'date'"):
            _split_df_into_time_chunks(df, 2)


class TestWithoutDfs:
    @pytest.fixture
    def tree(self, engine: BacktestEngine) -> tuple[Any, Any]:
        strategy = engine.add_strategy(MyStrategy(), "1d")
        feature = strategy.add_feature(MyFeature(), name="ema")
        return strategy, feature

    @staticmethod
    def _stores(*components: Any) -> list[Any]:
        return [
            store
            for component in components
            for store in [component.store, *component.data_stores.values()]
        ]

    def test_dfs_of_the_whole_tree_are_detached_and_restored(
        self, tree: tuple[Any, Any]
    ):
        stores = self._stores(*tree)
        dfs = [_wide_df(i + 1) for i in range(len(stores))]
        for store, df in zip(stores, dfs, strict=True):
            store._df = df

        with _without_dfs(tree[0]):
            assert all(store._df is None for store in stores)

        assert all(store._df is df for store, df in zip(stores, dfs, strict=True))

    def test_dfs_are_restored_on_error(self, tree: tuple[Any, Any]):
        stores = self._stores(*tree)
        dfs = [_wide_df(1) for _ in stores]
        for store, df in zip(stores, dfs, strict=True):
            store._df = df

        with pytest.raises(RuntimeError, match="boom"), _without_dfs(tree[0]):
            raise RuntimeError("boom")

        assert all(store._df is df for store, df in zip(stores, dfs, strict=True))


def _backtested(df: pl.DataFrame) -> pl.DataFrame:
    """A fake backtest: adds a column computed from the df, so the result tells which rows went in."""
    return df.with_columns((pl.col("x") * 2).alias("signal"))


@pytest.fixture
def calls() -> list[str]:
    """Records the order of the stubbed component steps."""
    return []


@pytest.fixture
def no_component_build(mocker: MockerFixture, calls: list[str]) -> Any:
    """Stubs the component side of build(): gathering and building need data and accounts,
    so build() and run() only exercise the engine's own flow."""
    mocker.patch.object(ComponentMixin, "start")
    mocker.patch.object(ComponentMixin, "stop")
    mocker.patch.object(
        BacktestMixin,
        "_gather",
        autospec=True,
        side_effect=lambda component: calls.append(f"gather {component.name}"),
    )
    return mocker.patch.object(
        BacktestMixin,
        "_build",
        autospec=True,
        side_effect=lambda component: calls.append(f"build {component.name}"),
    )


@pytest.fixture
def event_driven(engine: BacktestEngine, mocker: MockerFixture) -> None:
    mocker.patch.object(engine.context, "mode", BacktestMode.EVENT_DRIVEN)
    # the build is mocked, so no market data is loaded: an empty df makes the bar loop a no-op
    mocker.patch.object(DataFrameMixin, "get_df", return_value=pl.DataFrame())


@pytest.mark.usefixtures("no_component_build")
class TestBuild:
    def test_graph_is_validated_before_any_component_is_built(
        self, engine: BacktestEngine, mocker: MockerFixture, calls: list[str]
    ):
        engine.add_strategy(MyStrategy(), "1d", name="alpha")
        mocker.patch.object(
            engine,
            "_assert_unique_component_names",
            side_effect=lambda: calls.append("assert unique names"),
        )

        engine.build()

        assert calls == ["gather alpha", "assert unique names", "build alpha"]

    def test_build_does_not_start_the_engine(
        self, engine: BacktestEngine, mocker: MockerFixture
    ):
        strategy = engine.add_strategy(MyStrategy(), "1d", name="alpha")
        start = mocker.spy(strategy, "start")

        engine.build()

        assert not engine.is_running()
        start.assert_not_called()

    def test_second_build_is_a_no_op(self, engine: BacktestEngine, calls: list[str]):
        engine.add_strategy(MyStrategy(), "1d", name="alpha")

        engine.build()
        engine.build()

        assert calls == ["gather alpha", "build alpha"]

    def test_failed_build_can_be_retried(
        self, engine: BacktestEngine, no_component_build: Any, calls: list[str]
    ):
        engine.add_strategy(MyStrategy(), "1d", name="alpha")
        no_component_build.side_effect = RuntimeError("build failed")

        with pytest.raises(RuntimeError, match="build failed"):
            engine.build()

        no_component_build.side_effect = None
        engine.build()
        assert no_component_build.call_count == 2

    def test_use_split_before_build_is_rejected(self, engine: BacktestEngine):
        engine.add_strategy(MyStrategy(), "1d", name="alpha")

        with pytest.raises(RuntimeError, match=r"call engine.build\(\) first"):
            engine.use_split("train")


@pytest.mark.usefixtures("no_component_build")
class TestRun:
    def test_vectorized_run_builds_and_ends(
        self, engine: BacktestEngine, calls: list[str]
    ):
        engine.add_strategy(MyStrategy(), "1d", name="alpha")

        engine.run()

        assert calls == ["gather alpha", "build alpha"]
        assert not engine.is_running()

    @pytest.mark.usefixtures("event_driven")
    def test_run_builds_if_not_built(self, engine: BacktestEngine, calls: list[str]):
        engine.add_strategy(MyStrategy(), "1d", name="alpha")

        engine.run()

        assert calls == ["gather alpha", "build alpha"]

    @pytest.mark.usefixtures("event_driven")
    def test_run_does_not_rebuild_after_build(
        self, engine: BacktestEngine, calls: list[str]
    ):
        engine.add_strategy(MyStrategy(), "1d", name="alpha")

        engine.build()
        engine.run()

        assert calls == ["gather alpha", "build alpha"]

    @pytest.mark.usefixtures("event_driven")
    def test_engine_ends_after_a_successful_run(
        self, engine: BacktestEngine, mocker: MockerFixture
    ):
        engine.add_strategy(MyStrategy(), "1d", name="alpha")
        end = mocker.spy(engine, "end")

        engine.run()

        end.assert_called_once()
        assert not engine.is_running()

    @pytest.mark.usefixtures("event_driven")
    def test_build_error_is_logged_reraised_and_the_engine_ends(
        self, engine: BacktestEngine, mocker: MockerFixture, no_component_build: Any
    ):
        engine.add_strategy(MyStrategy(), "1d", name="alpha")
        no_component_build.side_effect = RuntimeError("build failed")
        log_exception = mocker.patch.object(engine._logger, "exception")

        with pytest.raises(RuntimeError, match="build failed"):
            engine.run()

        log_exception.assert_called_once()
        assert not engine.is_running()
        # the engine can run again
        no_component_build.side_effect = None
        engine.run()
        assert not engine.is_running()

    @pytest.mark.usefixtures("event_driven")
    def test_event_driven_dummy_strategy_without_model_or_feature_is_rejected(
        self, engine: BacktestEngine
    ):
        engine._create_dummy_strategy("1d")

        with pytest.raises(ValueError, match="No model or feature to backtest"):
            engine.run()

        assert not engine.is_running()


class FakeRay:
    """Stands in for Ray in chunked backtesting.

    A task runs eagerly when submitted, ray.wait() finishes the newest in-flight task first (so chunks finish out
    of order), ray.get() returns the task's result or raises its error.
    """

    def __init__(self) -> None:
        self._results: dict[object, tuple[bool, Any]] = {}
        self._chunk_nums: dict[object, int] = {}
        self.in_flight = 0
        self.max_in_flight = 0
        self.finish_order: list[int] = []
        self.cancelled: list[tuple[int, bool]] = []
        self.put_with_dfs_detached: list[bool] = []

    def remote(self, func: Callable[..., Any]) -> Any:
        fake = self

        class _Task:
            @staticmethod
            def remote(*args: Any) -> object:
                ref = object()
                fake._chunk_nums[ref] = args[-1]
                try:
                    fake._results[ref] = (True, func(*args))
                except Exception as exc:
                    fake._results[ref] = (False, exc)
                fake.in_flight += 1
                fake.max_in_flight = max(fake.max_in_flight, fake.in_flight)
                return ref

        return _Task()

    def put(self, obj: Any) -> Any:
        self.put_with_dfs_detached.append(obj.store._df is None)
        return obj

    def wait(
        self, refs: list[object], num_returns: int = 1
    ) -> tuple[list[object], list[object]]:
        assert num_returns == 1
        self.in_flight -= 1
        self.finish_order.append(self._chunk_nums[refs[-1]])
        return [refs[-1]], refs[:-1]

    def get(self, ref: object) -> Any:
        ok, value = self._results[ref]
        if not ok:
            raise value
        return value

    def cancel(self, ref: object, force: bool = False) -> None:
        self.cancelled.append((self._chunk_nums[ref], force))


@pytest.fixture
def fake_ray(mocker: MockerFixture) -> FakeRay:
    import pfund_kit.utils.ray as ray_utils
    import ray

    fake = FakeRay()
    for name in ("remote", "put", "wait", "get", "cancel"):
        mocker.patch.object(ray, name, getattr(fake, name))
    mocker.patch.object(ray_utils, "setup_ray")
    mocker.patch.object(
        ray_utils,
        "ray_logging_context",
        return_value=contextlib.nullcontext(mocker.Mock()),
    )
    mocker.patch.object(
        ray_utils, "setup_logger_in_ray_task", return_value=mocker.Mock()
    )
    return fake


def _stub_strategy_backtest(
    mocker: MockerFixture,
    strategy: Any,
    side_effect: Callable[[pl.DataFrame], pl.DataFrame] = _backtested,
) -> Any:
    """Backtests each chunk of strategy.backtest(X) by side_effect, tagged with its chunk_num like _backtest_chunk() does."""

    def backtest_chunk(df_chunk: pl.DataFrame, chunk_num: int = 0) -> pl.DataFrame:
        return side_effect(df_chunk).with_columns(pl.lit(chunk_num).alias(CHUNK_COL))

    return mocker.patch.object(strategy, "_backtest_chunk", side_effect=backtest_chunk)


class TestStrategyBacktest:
    @pytest.fixture
    def add_strategy(self, engine: BacktestEngine) -> Callable[..., Any]:
        """Adds the strategy with the given chunking kwargs (num_chunks, num_cpus)."""

        def add(**chunking: Any) -> Any:
            return engine.add_strategy(MyStrategy(), "1d", name="alpha", **chunking)

        return add

    def test_unchunked_backtest_is_chunk_0_of_X(
        self, add_strategy: Callable[..., Any], mocker: MockerFixture, fake_ray: FakeRay
    ):
        strategy = add_strategy()
        X = _wide_df(4)
        backtest = _stub_strategy_backtest(mocker, strategy)

        result = strategy.backtest(X)

        backtest.assert_called_once()
        # X goes into the user's backtest(df) as is
        assert backtest.call_args.args[0] is X
        assert result.get_column(CHUNK_COL).to_list() == [0] * 4
        assert result.get_column("signal").to_list() == [0, 2, 4, 6]
        # no Ray without chunking
        assert fake_ray.put_with_dfs_detached == []
        assert fake_ray.finish_order == []

    @pytest.mark.parametrize("num_chunks", [0, -1])
    def test_num_chunks_below_one_is_one(
        self,
        add_strategy: Callable[..., Any],
        mocker: MockerFixture,
        fake_ray: FakeRay,
        num_chunks: int,
    ):
        strategy = add_strategy(num_chunks=num_chunks)
        _stub_strategy_backtest(mocker, strategy)

        result = strategy.backtest(_wide_df(3))

        assert result.get_column(CHUNK_COL).to_list() == [0] * 3
        assert fake_ray.finish_order == []

    def test_chunks_are_combined_in_chunk_order(
        self, add_strategy: Callable[..., Any], mocker: MockerFixture, fake_ray: FakeRay
    ):
        strategy = add_strategy(num_chunks=4, num_cpus=2)
        df = _wide_df(10)
        _stub_strategy_backtest(mocker, strategy)

        result = strategy.backtest(df)

        # finished out of order, combined in order
        assert fake_ray.finish_order != sorted(fake_ray.finish_order)
        assert (
            result.get_column(CHUNK_COL).to_list()
            == [0] * 3 + [1] * 3 + [2] * 2 + [3] * 2
        )
        assert result.drop(CHUNK_COL).equals(_backtested(df))

    def test_each_chunk_is_backtested_on_its_own_date_range(
        self, add_strategy: Callable[..., Any], mocker: MockerFixture, fake_ray: FakeRay
    ):
        strategy = add_strategy(num_chunks=3)
        df = _long_df(6, ["BTC", "ETH"])
        backtest = _stub_strategy_backtest(mocker, strategy)

        strategy.backtest(df)

        backtested = sorted(
            (call.args[0] for call in backtest.call_args_list),
            key=lambda chunk: chunk.get_column("date").min(),
        )
        expected = [chunk for chunk, _ in _split_df_into_time_chunks(df, 3)]
        assert len(backtested) == len(expected)
        assert all(a.equals(b) for a, b in zip(backtested, expected, strict=True))

    def test_strategy_is_put_once_with_its_dfs_detached(
        self, add_strategy: Callable[..., Any], mocker: MockerFixture, fake_ray: FakeRay
    ):
        strategy = add_strategy(num_chunks=3)
        _stub_strategy_backtest(mocker, strategy)
        df = _wide_df(1)
        strategy.store._df = df

        strategy.backtest(_wide_df(6))

        assert fake_ray.put_with_dfs_detached == [True]
        # restored after ray.put()
        assert strategy.store._df is df

    @pytest.mark.parametrize(
        ("num_cpus", "cpu_count", "expected"),
        [(2, None, 2), (0, None, 1), (None, 3, 3), (None, None, 1), (10, None, 5)],
    )
    def test_num_cpus_caps_the_chunks_in_flight(
        self,
        add_strategy: Callable[..., Any],
        mocker: MockerFixture,
        fake_ray: FakeRay,
        num_cpus: int | None,
        cpu_count: int | None,
        expected: int,
    ):
        # num_cpus is resolved in add_strategy(), so os.cpu_count() is patched before it
        mocker.patch("os.cpu_count", return_value=cpu_count)
        strategy = add_strategy(num_chunks=5, num_cpus=num_cpus)
        _stub_strategy_backtest(mocker, strategy)

        result = strategy.backtest(_wide_df(10))

        assert fake_ray.max_in_flight == expected
        assert result.get_column(CHUNK_COL).unique().sort().to_list() == [0, 1, 2, 3, 4]

    def test_failed_chunk_cancels_the_chunks_in_flight(
        self, add_strategy: Callable[..., Any], mocker: MockerFixture, fake_ray: FakeRay
    ):
        strategy = add_strategy(num_chunks=3, num_cpus=2)

        def backtest(df: pl.DataFrame) -> pl.DataFrame:
            # chunk 1 of 3 (x = 2, 3) fails
            if 2 in df.get_column("x").to_list():
                raise ValueError("chunk failed")
            return _backtested(df)

        _stub_strategy_backtest(mocker, strategy, side_effect=backtest)

        with pytest.raises(ValueError, match="chunk failed"):
            strategy.backtest(_wide_df(6))

        # chunks 0 and 1 in flight, chunk 1 finishes first and fails, chunk 0 is force-cancelled, chunk 2 never starts
        assert fake_ray.finish_order == [1]
        assert fake_ray.cancelled == [(0, True)]


class _SplitModel(SKLearnModel):
    def predict(self, X: Any, *args: Any, **kwargs: Any) -> Any:
        return [0]


SPLIT_DATES = [
    datetime.datetime(2024, 1, 1) + datetime.timedelta(days=i) for i in range(10)
]


@pytest.fixture
def split_engine() -> BacktestEngine:
    # 721 over 10 daily bars: train 1/1-1/7, val 1/8-1/9, test 1/10
    return BacktestEngine(
        data_range=("2024-01-01", "2024-01-10"),
        settings=BacktestEngineSettings(),
        dataset_splits=721,
    )


def _seed_model(engine: BacktestEngine) -> Any:
    """A model as the backtestee with its trading df seeded, as if the engine had built it."""
    model = engine.add_model(_SplitModel(LinearRegression()), "1d", name="model")
    model.store.set_df(pl.DataFrame({"date": SPLIT_DATES, "x": list(range(10))}))
    engine._is_built = True
    return model


@pytest.fixture
def split_model(split_engine: BacktestEngine) -> Any:
    return _seed_model(split_engine)


def _x_values(model: Any) -> list[int]:
    return model.X.get_column("x").to_list()


class TestUseSplit:
    def test_sets_x_and_records_the_split(
        self, split_engine: BacktestEngine, split_model: Any
    ):
        split_engine.use_split("train")

        assert _x_values(split_model) == list(range(7))
        assert split_engine.context.active_split == ("train", None)

    def test_clear_split_restores_the_whole_range(
        self, split_engine: BacktestEngine, split_model: Any
    ):
        split_engine.use_split("test")
        split_engine.clear_split()

        assert _x_values(split_model) == list(range(10))
        assert split_engine.context.active_split is None

    def test_clear_split_without_a_split_is_a_no_op(
        self, split_engine: BacktestEngine, split_model: Any
    ):
        split_engine.clear_split()

        assert _x_values(split_model) == list(range(10))

    def test_switching_splits_maps_on_the_whole_range(
        self, split_engine: BacktestEngine, split_model: Any
    ):
        split_engine.use_split("test")
        split_engine.use_split("val")

        assert _x_values(split_model) == [7, 8]

    def test_context_manager_clears_on_exit(
        self, split_engine: BacktestEngine, split_model: Any
    ):
        with split_engine.use_split("train"):
            assert _x_values(split_model) == list(range(7))

        assert _x_values(split_model) == list(range(10))
        assert split_engine.context.active_split is None

    def test_context_manager_clears_on_error(
        self, split_engine: BacktestEngine, split_model: Any
    ):
        with pytest.raises(RuntimeError, match="boom"), split_engine.use_split("train"):
            raise RuntimeError("boom")

        assert _x_values(split_model) == list(range(10))
        assert split_engine.context.active_split is None


class TestApplySplit:
    def test_returns_the_rows_of_the_active_split(
        self, split_engine: BacktestEngine, split_model: Any
    ):
        y = pl.Series("y", [i * 10 for i in range(10)])

        with split_engine.use_split("val"):
            assert split_engine.apply_split(y).to_list() == [70, 80]

    def test_is_aligned_with_x(self, split_engine: BacktestEngine, split_model: Any):
        y = pl.Series("y", list(range(10)))

        with split_engine.use_split("train"):
            assert split_engine.apply_split(y).to_list() == _x_values(split_model)

    def test_requires_an_active_split(
        self, split_engine: BacktestEngine, split_model: Any
    ):
        with pytest.raises(RuntimeError, match=r"call engine.use_split\(\) before"):
            split_engine.apply_split(pl.Series("y", list(range(10))))

    def test_rejects_data_aligned_with_the_split_x(
        self, split_engine: BacktestEngine, split_model: Any
    ):
        with (
            split_engine.use_split("train"),
            pytest.raises(ValueError, match="row-aligned"),
        ):
            split_engine.apply_split(pl.Series("y", list(range(7))))


class TestTrainedOn:
    def test_stamps_the_active_split(
        self, split_engine: BacktestEngine, split_model: Any
    ):
        with split_engine.use_split("train"):
            trained_on = split_model._get_trained_on()

        assert trained_on.part == "train"
        assert trained_on.fold is None
        assert trained_on.intervals == [(SPLIT_DATES[0], SPLIT_DATES[7])]

    def test_trainval_is_allowed(self, split_engine: BacktestEngine, split_model: Any):
        with split_engine.use_split("trainval"):
            trained_on = split_model._get_trained_on()

        assert trained_on.intervals == [(SPLIT_DATES[0], SPLIT_DATES[9])]

    def test_no_split_in_use_is_rejected(self, split_model: Any):
        with pytest.raises(RuntimeError, match="no split in use"):
            split_model._get_trained_on()

    @pytest.mark.parametrize("part", ["val", "test"])
    def test_val_and_test_are_rejected(
        self, split_engine: BacktestEngine, split_model: Any, part: Any
    ):
        with (
            split_engine.use_split(part),
            pytest.raises(ValueError, match=f"trained on the {part} set"),
        ):
            split_model._get_trained_on()

    def test_without_dataset_splits_it_is_the_whole_range(self):
        engine = BacktestEngine(
            data_range=("2024-01-01", "2024-01-10"), settings=BacktestEngineSettings()
        )
        model = _seed_model(engine)

        trained_on = model._get_trained_on()

        assert trained_on.part is None
        assert trained_on.intervals == [
            (SPLIT_DATES[0], SPLIT_DATES[-1] + datetime.timedelta(days=1))
        ]
