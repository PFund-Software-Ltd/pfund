"""Tests the add-component flow under the backtest engine: engine.add_strategy/add_model/add_feature
(top-level, hosted by the dummy strategy for models/features) and component.add_* (adopted children)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

import pytest
from pfund_blocks.resolution import Resolution
from sklearn.linear_model import LinearRegression

import pfund as pf
from pfund.backtest._mixins.feature_mixin import BacktestFeatureMixin
from pfund.backtest._mixins.model_mixin import BacktestModelMixin
from pfund.backtest._mixins.strategy_mixin import BacktestStrategyMixin
from pfund.components.actor_proxy import ActorProxy
from pfund.components.models.sklearn_model import SKLearnModel
from pfund.components.strategies._dummy_strategy import _DummyStrategy
from pfund.engines.backtest_engine import BacktestEngine
from pfund.engines.settings.backtest_engine_settings import BacktestEngineSettings
from pfund.enums import RunMode


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
def create_actor(mocker: MockerFixture) -> Any:
    return mocker.patch.object(ActorProxy, "_create_actor")


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
