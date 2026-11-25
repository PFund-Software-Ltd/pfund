"""Tests BacktestEngine: construction, the add-component and add-data flows, and run(), i.e. picking the backtestee,
chunking the df (sequentially or over Ray), and post-processing, persisting and returning the backtest df.

The backtestee's own backtest (BacktestMixin._vectorized_backtest/_event_driven_backtest) and the component setup
(gather/start/stop) are stubbed in the run() tests, only the engine's orchestration is tested here.
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
from pfund.backtest._mixins.feature_mixin import BacktestFeatureMixin
from pfund.backtest._mixins.model_mixin import BacktestModelMixin
from pfund.backtest._mixins.strategy_mixin import BacktestStrategyMixin
from pfund.backtest.cv.holdout import Holdout
from pfund.components._mixins.component_mixin import ComponentMixin
from pfund.components.actor_proxy import ActorProxy
from pfund.components.models.sklearn_model import SKLearnModel
from pfund.components.strategies._dummy_strategy import _DummyStrategy
from pfund.data_config import DataConfig
from pfund.engines.backtest_engine import (
    BacktestEngine,
    _split_df_into_time_chunks,
    _without_dfs,
)
from pfund.engines.settings.backtest_engine_settings import BacktestEngineSettings
from pfund.enums import BacktestMode, RunMode
from pfund.simulated import SimulatedVenue

VENUE = "BYBIT"
BTC = "BTC_USDT_PERP"
ETH = "ETH_USDT_PERP"
BTC_NAME = "BYBIT_BTC_USDT_PERPETUAL"
CHUNK_COL = BacktestEngine.CHUNK_COL


class MyStrategy(pf.Strategy):
    def trade(self, X: Any) -> list[Any]:
        return []


class MyFeature(pf.Feature):
    def transform(self, X: Any, *args: Any, **kwargs: Any) -> Any:
        return [0]


def _is_top_component(component: Any) -> bool:
    return component._is_top_component


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
        assert engine.context.num_chunks == 1
        assert engine.context.dataset_splits is None
        assert engine.result is None
        assert not engine.is_running()

    def test_mode_accepts_a_string(self):
        engine = BacktestEngine(settings=BacktestEngineSettings(), mode="vectorized")

        assert engine.backtest_mode == BacktestMode.VECTORIZED

    def test_dataset_splits_is_turned_into_a_recipe(self):
        engine = BacktestEngine(settings=BacktestEngineSettings(), dataset_splits=721)

        assert isinstance(engine.context.dataset_splits, Holdout)


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
        assert _is_top_component(strategy)
        assert engine.get_strategy("alpha") is strategy

    def test_ray_args_are_ignored_with_a_warning(
        self, engine: BacktestEngine, mocker: MockerFixture, create_actor: Any
    ):
        warning = mocker.patch.object(engine._logger, "warning")
        strategy = MyStrategy()

        added = engine.add_strategy(
            strategy, "1d", ray_actor_options={"name": "x"}, num_cpus=1
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
        assert _is_top_component(model)

    def test_feature_is_hosted_by_dummy_strategy(self, engine: BacktestEngine):
        feature = MyFeature()

        added = engine.add_feature(feature, "1d", name="ema", df_form="long")

        dummy = engine.get_strategy(_DummyStrategy.__name__)
        assert added is feature
        assert isinstance(feature, BacktestFeatureMixin)
        assert feature.name == "ema"
        assert feature.df_form == "long"
        assert dummy.features == {"ema": feature}
        assert _is_top_component(feature)

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
        assert not _is_top_component(feature)

    def test_child_model_is_wrapped_and_adopted(self, strategy: MyStrategy):
        model = strategy.add_model(LinearRegression())

        assert isinstance(model, SKLearnModel)
        assert isinstance(model, BacktestModelMixin)
        assert strategy.models == {"LinearRegression": model}
        assert not _is_top_component(model)

    def test_substrategy_is_adopted_and_marked(self, strategy: MyStrategy):
        sub = MyStrategy()

        added = strategy.add_strategy(sub, name="sub")

        assert added is sub
        assert isinstance(sub, BacktestStrategyMixin)
        assert sub.is_substrategy()
        assert strategy.strategies == {"sub": sub}
        assert not _is_top_component(sub)

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

    def test_as_features_false_is_kept(self, strategy: Any):
        strategy.add_data(VENUE, BTC, as_features=False)

        assert strategy.market_data_store.data_as_features is False

    def test_inconsistent_as_features_is_rejected(self, strategy: Any):
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


def _stub_backtest(
    mocker: MockerFixture,
    component: Any,
    df: pl.DataFrame,
    *,
    df_kind: str = "full_df",
    side_effect: Callable[[pl.DataFrame], pl.DataFrame] | Exception = _backtested,
    method: str = "_vectorized_backtest",
) -> Any:
    """Makes `component.{df_kind}` return df, and the other df kinds raise, then stubs its backtest method."""
    for kind in ("data_df", "trading_df", "full_df"):
        if kind == df_kind:
            prop = mocker.PropertyMock(return_value=df)
        else:
            prop = mocker.PropertyMock(
                side_effect=AssertionError(f"{kind} must not be read")
            )
        mocker.patch.object(type(component), kind, new_callable=lambda prop=prop: prop)
    return mocker.patch.object(component, method, side_effect=side_effect)


@pytest.fixture
def no_component_setup(engine: BacktestEngine, mocker: MockerFixture) -> Any:
    """Skips the component setup in run(): gathering needs data and accounts (the BacktestMixin side),
    so run() only exercises the engine's own flow."""
    mocker.patch.object(ComponentMixin, "start")
    mocker.patch.object(ComponentMixin, "stop")
    return mocker.patch.object(engine, "_setup")


@pytest.mark.usefixtures("no_component_setup")
class TestRunStrategy:
    @pytest.fixture
    def strategy(self, engine: BacktestEngine) -> Any:
        return engine.add_strategy(MyStrategy(), "1d", name="alpha")

    @pytest.fixture
    def persist(self, strategy: Any, mocker: MockerFixture) -> Any:
        return mocker.patch.object(strategy.store, "persist_to_datalake")

    def test_full_df_is_backtested_and_returned_as_chunk_0(
        self, engine: BacktestEngine, strategy: Any, mocker: MockerFixture, persist: Any
    ):
        df = _wide_df(4)
        backtest = _stub_backtest(mocker, strategy, df)

        result = engine.run()

        backtest.assert_called_once()
        assert backtest.call_args.args[0].equals(df)
        assert result is not None
        assert result.get_column(CHUNK_COL).to_list() == [0] * 4
        assert result.get_column("signal").to_list() == [0, 2, 4, 6]
        assert engine.result is result

    def test_result_is_persisted_once_in_vectorized_mode(
        self, engine: BacktestEngine, strategy: Any, mocker: MockerFixture, persist: Any
    ):
        _stub_backtest(mocker, strategy, _wide_df(3))

        result = engine.run()

        persist.assert_called_once_with()
        assert strategy.store._df is result

    def test_event_driven_backtests_data_df_and_does_not_persist(
        self, engine: BacktestEngine, strategy: Any, mocker: MockerFixture, persist: Any
    ):
        mocker.patch.object(engine.context, "mode", BacktestMode.EVENT_DRIVEN)
        df = _wide_df(3)
        backtest = _stub_backtest(
            mocker, strategy, df, df_kind="data_df", method="_event_driven_backtest"
        )

        result = engine.run()

        assert backtest.call_args.args[0].equals(df)
        assert result is not None
        assert result.get_column(CHUNK_COL).to_list() == [0] * 3
        persist.assert_not_called()

    def test_num_chunks_is_set_before_the_components_are_set_up(
        self,
        engine: BacktestEngine,
        strategy: Any,
        mocker: MockerFixture,
        persist: Any,
        no_component_setup: Any,
        fake_ray: FakeRay,
    ):
        _stub_backtest(mocker, strategy, _wide_df(3))
        num_chunks_at_setup: list[int] = []
        no_component_setup.side_effect = lambda: num_chunks_at_setup.append(
            engine.context.num_chunks
        )

        engine.run(num_chunks=3)

        assert num_chunks_at_setup == [3]

    @pytest.mark.parametrize("num_chunks", [0, -1])
    def test_num_chunks_below_one_is_one(
        self,
        engine: BacktestEngine,
        strategy: Any,
        mocker: MockerFixture,
        persist: Any,
        num_chunks: int,
    ):
        _stub_backtest(mocker, strategy, _wide_df(3))

        result = engine.run(num_chunks=num_chunks)

        assert engine.context.num_chunks == 1
        assert result is not None
        assert result.get_column(CHUNK_COL).to_list() == [0] * 3

    def test_error_is_logged_reraised_and_the_engine_ends(
        self, engine: BacktestEngine, strategy: Any, mocker: MockerFixture, persist: Any
    ):
        _stub_backtest(
            mocker, strategy, _wide_df(3), side_effect=ValueError("bad backtest")
        )
        log_exception = mocker.patch.object(engine._logger, "exception")

        with pytest.raises(ValueError, match="bad backtest"):
            engine.run()

        log_exception.assert_called_once()
        assert not engine.is_running()
        assert engine.result is None
        persist.assert_not_called()

    def test_engine_ends_after_a_successful_run(
        self, engine: BacktestEngine, strategy: Any, mocker: MockerFixture, persist: Any
    ):
        _stub_backtest(mocker, strategy, _wide_df(3))
        end = mocker.spy(engine, "end")

        engine.run()

        end.assert_called_once()
        assert not engine.is_running()

    def test_engine_ends_when_the_component_setup_fails(
        self,
        engine: BacktestEngine,
        strategy: Any,
        mocker: MockerFixture,
        persist: Any,
        no_component_setup: Any,
    ):
        _stub_backtest(mocker, strategy, _wide_df(3))
        no_component_setup.side_effect = RuntimeError("setup failed")

        with pytest.raises(RuntimeError, match="setup failed"):
            engine.run()

        assert not engine.is_running()
        # the engine can run again
        no_component_setup.side_effect = None
        assert engine.run() is not None


@pytest.mark.usefixtures("no_component_setup")
class TestRunFeature:
    @pytest.fixture
    def feature(self, engine: BacktestEngine, mocker: MockerFixture) -> Any:
        feature = engine.add_feature(MyFeature(), "1d", name="ema")
        mocker.patch.object(feature.store, "persist_to_datalake")
        return feature

    def test_trading_df_is_backtested_without_chunk_col(
        self, engine: BacktestEngine, feature: Any, mocker: MockerFixture
    ):
        df = _wide_df(3)
        backtest = _stub_backtest(mocker, feature, df, df_kind="trading_df")

        result = engine.run()

        assert backtest.call_args.args[0].equals(df)
        assert result is not None
        assert result.equals(_backtested(df))

    def test_result_is_persisted_to_the_feature_store(
        self, engine: BacktestEngine, feature: Any, mocker: MockerFixture
    ):
        _stub_backtest(mocker, feature, _wide_df(3), df_kind="trading_df")
        dummy = engine.get_strategy(_DummyStrategy.__name__)
        dummy_persist = mocker.patch.object(dummy.store, "persist_to_datalake")

        result = engine.run()

        feature.store.persist_to_datalake.assert_called_once_with()
        assert feature.store._df is result
        dummy_persist.assert_not_called()

    def test_chunking_falls_back_to_one_chunk_with_a_warning(
        self, engine: BacktestEngine, feature: Any, mocker: MockerFixture
    ):
        backtest = _stub_backtest(mocker, feature, _wide_df(4), df_kind="trading_df")
        warning = mocker.patch.object(engine._logger, "warning")

        engine.run(num_chunks=2)

        assert engine.context.num_chunks == 1
        backtest.assert_called_once()
        assert any(
            "only supported for strategy backtesting" in call.args[0]
            for call in warning.call_args_list
        )


@pytest.mark.usefixtures("no_component_setup")
class TestRunModel:
    @pytest.fixture
    def model(self, engine: BacktestEngine, mocker: MockerFixture) -> Any:
        model = engine.add_model(LinearRegression(), "1d", name="lr")
        mocker.patch.object(model.store, "persist_to_datalake")
        return model

    def test_training_returns_none_without_backtesting(
        self, engine: BacktestEngine, model: Any, mocker: MockerFixture
    ):
        model._is_training = True
        backtest = _stub_backtest(mocker, model, _wide_df(3), df_kind="trading_df")

        result = engine.run()

        assert result is None
        assert engine.result is None
        backtest.assert_not_called()
        model.store.persist_to_datalake.assert_not_called()
        assert not engine.is_running()

    def test_event_driven_training_is_rejected(
        self, engine: BacktestEngine, model: Any, mocker: MockerFixture
    ):
        mocker.patch.object(engine.context, "mode", BacktestMode.EVENT_DRIVEN)
        model._is_training = True

        with pytest.raises(ValueError, match="needs a trained model"):
            engine.run()

        assert not engine.is_running()

    def test_trained_model_is_backtested(
        self, engine: BacktestEngine, model: Any, mocker: MockerFixture
    ):
        df = _wide_df(3)
        backtest = _stub_backtest(mocker, model, df, df_kind="trading_df")

        result = engine.run()

        assert backtest.call_args.args[0].equals(df)
        assert result is not None
        assert result.equals(_backtested(df))
        assert CHUNK_COL not in result.columns


@pytest.mark.usefixtures("no_component_setup")
def test_run_without_a_strategy_is_rejected(
    engine: BacktestEngine, no_component_setup: Any
):
    with pytest.raises(ValueError, match="Nothing to backtest"):
        engine.run()

    assert not engine.is_running()
    no_component_setup.assert_not_called()


@pytest.mark.usefixtures("no_component_setup")
def test_dummy_strategy_without_model_or_feature_is_rejected(engine: BacktestEngine):
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


@pytest.mark.usefixtures("no_component_setup")
class TestRunChunked:
    @pytest.fixture
    def strategy(self, engine: BacktestEngine, mocker: MockerFixture) -> Any:
        strategy = engine.add_strategy(MyStrategy(), "1d", name="alpha")
        mocker.patch.object(strategy.store, "persist_to_datalake")
        return strategy

    def test_chunks_are_combined_in_chunk_order(
        self,
        engine: BacktestEngine,
        strategy: Any,
        mocker: MockerFixture,
        fake_ray: FakeRay,
    ):
        df = _wide_df(10)
        _stub_backtest(mocker, strategy, df)

        result = engine.run(num_chunks=4, num_cpus=2)

        # finished out of order, combined in order
        assert fake_ray.finish_order != sorted(fake_ray.finish_order)
        assert result is not None
        assert (
            result.get_column(CHUNK_COL).to_list()
            == [0] * 3 + [1] * 3 + [2] * 2 + [3] * 2
        )
        assert result.drop(CHUNK_COL).equals(_backtested(df))
        assert strategy.store._df is result
        strategy.store.persist_to_datalake.assert_called_once_with()

    def test_each_chunk_is_backtested_on_its_own_date_range(
        self,
        engine: BacktestEngine,
        strategy: Any,
        mocker: MockerFixture,
        fake_ray: FakeRay,
    ):
        df = _long_df(6, ["BTC", "ETH"])
        backtest = _stub_backtest(mocker, strategy, df)

        engine.run(num_chunks=3)

        backtested = sorted(
            (call.args[0] for call in backtest.call_args_list),
            key=lambda chunk: chunk.get_column("date").min(),
        )
        expected = [chunk for chunk, _ in _split_df_into_time_chunks(df, 3)]
        assert len(backtested) == len(expected)
        assert all(a.equals(b) for a, b in zip(backtested, expected, strict=True))

    def test_backtestee_is_put_once_with_its_dfs_detached(
        self,
        engine: BacktestEngine,
        strategy: Any,
        mocker: MockerFixture,
        fake_ray: FakeRay,
    ):
        _stub_backtest(mocker, strategy, _wide_df(6))
        df = _wide_df(1)
        strategy.store.set_df(df)

        engine.run(num_chunks=3)

        assert fake_ray.put_with_dfs_detached == [True]

    @pytest.mark.parametrize(
        ("num_cpus", "cpu_count", "expected"),
        [(2, None, 2), (0, None, 1), (None, 3, 3), (None, None, 1), (10, None, 5)],
    )
    def test_num_cpus_caps_the_chunks_in_flight(
        self,
        engine: BacktestEngine,
        strategy: Any,
        mocker: MockerFixture,
        fake_ray: FakeRay,
        num_cpus: int | None,
        cpu_count: int | None,
        expected: int,
    ):
        mocker.patch("os.cpu_count", return_value=cpu_count)
        _stub_backtest(mocker, strategy, _wide_df(10))

        result = engine.run(num_chunks=5, num_cpus=num_cpus)

        assert fake_ray.max_in_flight == expected
        assert result is not None
        assert result.get_column(CHUNK_COL).unique().sort().to_list() == [0, 1, 2, 3, 4]

    def test_failed_chunk_cancels_the_chunks_in_flight(
        self,
        engine: BacktestEngine,
        strategy: Any,
        mocker: MockerFixture,
        fake_ray: FakeRay,
    ):
        def backtest(df: pl.DataFrame) -> pl.DataFrame:
            # chunk 1 of 3 (x = 2, 3) fails
            if 2 in df.get_column("x").to_list():
                raise ValueError("chunk failed")
            return _backtested(df)

        _stub_backtest(mocker, strategy, _wide_df(6), side_effect=backtest)

        with pytest.raises(ValueError, match="chunk failed"):
            engine.run(num_chunks=3, num_cpus=2)

        # chunks 0 and 1 in flight, chunk 1 finishes first and fails, chunk 0 is force-cancelled, chunk 2 never starts
        assert fake_ray.finish_order == [1]
        assert fake_ray.cancelled == [(0, True)]
        assert engine.result is None
        strategy.store.persist_to_datalake.assert_not_called()
        assert not engine.is_running()

    def test_unchunked_run_does_not_use_ray(
        self,
        engine: BacktestEngine,
        strategy: Any,
        mocker: MockerFixture,
        fake_ray: FakeRay,
    ):
        _stub_backtest(mocker, strategy, _wide_df(3))

        engine.run(num_chunks=1)

        assert fake_ray.put_with_dfs_detached == []
        assert fake_ray.finish_order == []
