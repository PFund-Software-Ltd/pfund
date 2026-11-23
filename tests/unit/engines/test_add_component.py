"""Tests the add-component flow outside backtesting: engine.add_strategy and
component.add_strategy/add_model/add_feature, local and remote (ActorProxy), and adding
children through a remote parent (ActorProxy.add_* -> _add_component_via_proxy).

Backtesting's own flow (BacktestMixin._add_component) is tested in test_backtest_engine.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, assert_type

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

    from pfund.components.models.model_base import BaseModel

import pytest
from pfund_blocks.resolution import Resolution
from sklearn.linear_model import LinearRegression

import pfund as pf
from pfund.components.actor_proxy import ActorProxy
from pfund.components.models.sklearn_model import SKLearnModel
from pfund.engines.settings.trade_engine_settings import TradeEngineSettings
from pfund.engines.trade_engine import TradeEngine
from pfund.enums import ComponentType, RunMode


class MyStrategy(pf.Strategy):
    def trade(self, X: Any) -> list[Any]:
        return []


class MyFeature(pf.Feature):
    def transform(self, X: Any, *args: Any, **kwargs: Any) -> Any:
        return [0]


class MyModel(pf.Model):
    def predict(self, X: Any, *args: Any, **kwargs: Any) -> Any:
        return [0]

    def _encode(self, payload: dict[str, Any]) -> bytes:
        return b""

    def _decode(self, data: bytes) -> None:
        pass


@pytest.fixture
def engine() -> TradeEngine:
    # NOTE: PAPER, not the default SANDBOX, ENGINE_SETTINGS has no SANDBOX entry yet
    return TradeEngine(env="PAPER", settings=TradeEngineSettings())


@pytest.fixture
def create_actor(mocker: MockerFixture) -> Any:
    """Patches out Ray actor creation, so tests can check whether the add flow created an actor."""
    return mocker.patch.object(ActorProxy, "_create_actor")


@pytest.fixture
def ray_get(mocker: MockerFixture) -> Any:
    """Lets a proxy call its (fake) actor: no Ray startup, ray.get returns the value as is."""
    mocker.patch("pfund_kit.utils.ray.setup_ray")
    return mocker.patch("ray.get", side_effect=lambda ref: ref)


class TestEngineAddStrategy:
    def test_local_strategy_is_hydrated_and_registered(self, engine: TradeEngine):
        strategy = MyStrategy()

        added = engine.add_strategy(strategy, "1m")

        assert added is strategy
        assert strategy.name == "MyStrategy"
        assert strategy.run_mode == RunMode.LOCAL
        assert strategy.resolution == Resolution("1m")
        assert strategy.df_form == "long"
        assert strategy.context is engine.context
        assert engine.get_strategy(strategy.name) is strategy

    def test_explicit_name_is_used(self, engine: TradeEngine):
        strategy = engine.add_strategy(MyStrategy(), "1m", name="alpha")

        assert strategy.name == "alpha"
        assert engine.get_strategy("alpha") is strategy

    @pytest.mark.parametrize("second_name", ["alpha", "ALPHA"])
    def test_duplicate_name_is_rejected(self, engine: TradeEngine, second_name: str):
        engine.add_strategy(MyStrategy(), "1m", name="alpha")

        with pytest.raises(ValueError, match="already exists"):
            engine.add_strategy(MyStrategy(), "1m", name=second_name)

    def test_duplicate_default_name_is_rejected(self, engine: TradeEngine):
        engine.add_strategy(MyStrategy(), "1m")

        with pytest.raises(ValueError, match="already exists"):
            engine.add_strategy(MyStrategy(), "1m")

    def test_non_strategy_is_rejected(self, engine: TradeEngine):
        with pytest.raises(AssertionError, match="not an instance of BaseStrategy"):
            engine.add_strategy(MyFeature(), "1m")  # ty: ignore[invalid-argument-type]

    def test_ray_actor_options_alone_keeps_strategy_local(self, engine: TradeEngine):
        strategy = MyStrategy()

        added = engine.add_strategy(strategy, "1m", ray_actor_options={"name": "x"})

        assert added is strategy
        assert strategy.run_mode == RunMode.LOCAL


class TestEngineAddRemoteStrategy:
    def test_ray_kwargs_return_an_unstarted_proxy(
        self, engine: TradeEngine, create_actor: Any
    ):
        proxy = engine.add_strategy(MyStrategy(), "1m", name="alpha", num_cpus=1)

        assert isinstance(proxy, ActorProxy)
        assert proxy.name == "alpha"
        assert proxy.component_type == ComponentType.strategy
        assert proxy.resolution == Resolution("1m")
        assert proxy._hydrate_fields == {
            "name": "alpha",
            "run_mode": RunMode.REMOTE,
            "resolution": "1m",
            "engine_context": engine.context,
            "df_form": "long",
        }
        assert engine.get_strategy("alpha") is proxy
        create_actor.assert_not_called()

    def test_ray_actor_options_get_the_proxy_name(
        self, engine: TradeEngine, create_actor: Any
    ):
        proxy = engine.add_strategy(
            MyStrategy(), "1m", ray_actor_options={"max_restarts": 1}, num_cpus=1
        )

        assert isinstance(proxy, ActorProxy)
        assert proxy._ray_actor_options == {
            "max_restarts": 1,
            "name": "MyStrategy",
        }
        assert proxy._ray_kwargs == {"num_cpus": 1}

    def test_ray_kwargs_without_num_cpus_are_rejected(self, engine: TradeEngine):
        with pytest.raises(ValueError, match="`num_cpus` must be set"):
            engine.add_strategy(MyStrategy(), "1m", num_gpus=1)

    def test_non_positive_num_cpus_is_rejected(self, engine: TradeEngine):
        with pytest.raises(ValueError, match="`num_cpus` must be greater than 0"):
            engine.add_strategy(MyStrategy(), "1m", num_cpus=0)


class TestComponentAddLocal:
    @pytest.fixture
    def strategy(self, engine: TradeEngine) -> MyStrategy:
        strategy = MyStrategy()
        engine.add_strategy(strategy, "1m")
        return strategy

    def test_feature_is_hydrated_with_parent_context(
        self, engine: TradeEngine, strategy: MyStrategy
    ):
        feature = MyFeature()

        added = strategy.add_feature(feature, "1m", name="ema")

        assert added is feature
        assert feature.name == "ema"
        assert feature.run_mode == RunMode.LOCAL
        assert feature.resolution == Resolution("1m")
        assert feature.df_form == "wide"
        assert feature.context is engine.context
        assert strategy.features == {"ema": feature}

    def test_resolution_is_inherited_when_omitted(self, strategy: MyStrategy):
        feature = strategy.add_feature(MyFeature())

        assert feature is not None
        assert feature.resolution == strategy.resolution

    def test_coarser_child_resolution_is_allowed(self, strategy: MyStrategy):
        feature = strategy.add_feature(MyFeature(), "1d")

        assert feature is not None
        assert feature.resolution == Resolution("1d")

    def test_finer_child_resolution_is_rejected(self, engine: TradeEngine):
        strategy = engine.add_strategy(MyStrategy(), "1d")

        with pytest.raises(ValueError, match="is finer than parent"):
            strategy.add_feature(MyFeature(), "1m")

    def test_df_form_is_passed_through(self, strategy: MyStrategy):
        feature = strategy.add_feature(MyFeature(), df_form="long")

        assert feature is not None
        assert feature.df_form == "long"

    def test_model_is_added(self, strategy: MyStrategy):
        model = MyModel(LinearRegression())

        added = strategy.add_model(model, name="clf")

        assert added is model
        assert model.name == "clf"
        assert strategy.models == {"clf": model}

    def test_underlying_sklearn_model_is_wrapped(self, strategy: MyStrategy):
        model = strategy.add_model(LinearRegression())

        assert isinstance(model, SKLearnModel)
        assert model.name == "LinearRegression"
        assert strategy.models == {"LinearRegression": model}

    def test_unsupported_model_is_rejected(self, strategy: MyStrategy):
        with pytest.raises(TypeError, match="Unsupported model type"):
            strategy.add_model(object())  # ty: ignore[no-matching-overload]

    @pytest.mark.parametrize("second_name", ["ema", "EMA"])
    def test_duplicate_name_is_rejected(self, strategy: MyStrategy, second_name: str):
        first = strategy.add_feature(MyFeature(), name="ema")

        with pytest.raises(ValueError, match="already exists"):
            strategy.add_feature(MyFeature(), name=second_name)
        assert strategy.features == {"ema": first}

    def test_same_name_for_different_component_types_is_allowed(
        self, strategy: MyStrategy
    ):
        strategy.add_feature(MyFeature(), name="x")
        strategy.add_model(MyModel(LinearRegression()), name="x")

        assert list(strategy.features) == ["x"]
        assert list(strategy.models) == ["x"]

    def test_feature_can_add_child_feature(self, strategy: MyStrategy):
        parent = strategy.add_feature(MyFeature(), name="parent")
        assert parent is not None

        child = parent.add_feature(MyFeature(), name="child")

        assert parent.features == {"child": child}

    def test_child_of_a_component_in_its_actor_is_returned(
        self, strategy: MyStrategy, mocker: MockerFixture
    ):
        # simulates the strategy running inside its Ray actor, the child lives in the same process,
        # only adding through the proxy (ActorProxy.add_*) returns None
        mocker.patch.object(strategy, "_run_mode", RunMode.REMOTE)
        feature = MyFeature()

        added = strategy.add_feature(feature, name="ema")

        assert added is feature
        assert strategy.features == {"ema": feature}


class TestComponentAddSubStrategy:
    @pytest.fixture
    def strategy(self, engine: TradeEngine) -> MyStrategy:
        strategy = MyStrategy()
        engine.add_strategy(strategy, "1m")
        return strategy

    def test_substrategy_is_marked_and_registered(self, strategy: MyStrategy):
        sub = MyStrategy()

        added = strategy.add_strategy(sub, name="sub")

        assert added is sub
        assert sub.name == "sub"
        assert sub.is_substrategy()
        assert not strategy.is_substrategy()
        assert sub.df_form == "long"
        assert strategy.strategies == {"sub": sub}

    def test_substrategy_can_add_substrategy(self, strategy: MyStrategy):
        sub = strategy.add_strategy(MyStrategy(), name="sub")
        assert sub is not None

        subsub = sub.add_strategy(MyStrategy(), name="subsub")

        assert subsub is not None
        assert subsub.is_substrategy()
        assert sub.strategies == {"subsub": subsub}

    def test_only_a_strategy_can_add_a_substrategy(self, strategy: MyStrategy):
        feature = strategy.add_feature(MyFeature())
        assert feature is not None

        with pytest.raises(
            AssertionError, match="only a strategy can add a sub-strategy"
        ):
            feature._add_component(
                component=MyStrategy(), resolution="", name="", df_form="long"
            )

    def test_duplicate_substrategy_name_is_rejected(self, strategy: MyStrategy):
        strategy.add_strategy(MyStrategy(), name="sub")

        with pytest.raises(ValueError, match="already exists"):
            strategy.add_strategy(MyStrategy(), name="sub")


class TestComponentAddRemote:
    @pytest.fixture
    def strategy(self, engine: TradeEngine) -> MyStrategy:
        strategy = MyStrategy()
        engine.add_strategy(strategy, "1m")
        return strategy

    @pytest.fixture
    def feature_actor(self, create_actor: Any, ray_get: Any) -> Any:
        actor = create_actor.return_value
        actor.is_strategy.remote.return_value = False
        actor.is_model.remote.return_value = False
        actor.is_feature.remote.return_value = True
        return actor

    def test_ray_kwargs_return_an_unstarted_proxy(
        self, engine: TradeEngine, strategy: MyStrategy, create_actor: Any
    ):
        proxy = strategy.add_feature(MyFeature(), name="ema", num_cpus=1)

        assert isinstance(proxy, ActorProxy)
        assert proxy.name == "ema"
        assert proxy.component_type == ComponentType.feature
        assert proxy.resolution == strategy.resolution
        assert proxy._hydrate_fields == {
            "name": "ema",
            "run_mode": RunMode.REMOTE,
            "resolution": strategy.resolution,
            "engine_context": engine.context,
            "df_form": "wide",
        }
        assert strategy.features == {"ema": proxy}
        create_actor.assert_not_called()

    def test_remote_model_is_wrapped_before_proxying(
        self, strategy: MyStrategy, create_actor: Any
    ):
        proxy = strategy.add_model(LinearRegression(), num_cpus=1)

        assert isinstance(proxy, ActorProxy)
        assert isinstance(proxy._component, SKLearnModel)
        assert proxy.name == "LinearRegression"
        assert strategy.models == {"LinearRegression": proxy}
        create_actor.assert_not_called()

    def test_existing_proxy_keeps_its_name_and_resolution(
        self,
        engine: TradeEngine,
        strategy: MyStrategy,
        create_actor: Any,
        feature_actor: Any,
    ):
        other = engine.add_strategy(MyStrategy(), "1m", name="other")
        proxy = strategy.add_feature(MyFeature(), name="ema", num_cpus=1)
        assert isinstance(proxy, ActorProxy)
        hydrate_fields = proxy._hydrate_fields
        assert hydrate_fields is not None

        added = other.add_feature(proxy, "1d", name="ignored")

        assert added is proxy
        assert other.features == {"ema": proxy}
        assert proxy.resolution == Resolution("1m")
        assert proxy._hydrate_fields is hydrate_fields
        # the proxy's component type is read as a plain attribute, so the actor stays lazy
        create_actor.assert_not_called()

    def test_existing_proxy_finer_than_parent_is_rejected(
        self, engine: TradeEngine, strategy: MyStrategy, feature_actor: Any
    ):
        coarse = engine.add_strategy(MyStrategy(), "1d", name="coarse")
        proxy = strategy.add_feature(MyFeature(), num_cpus=1)
        assert isinstance(proxy, ActorProxy)

        with pytest.raises(ValueError, match="is finer than parent"):
            coarse.add_feature(proxy)

    def test_remote_substrategy_is_marked_without_creating_the_actor(
        self, strategy: MyStrategy, create_actor: Any
    ):
        proxy = strategy.add_strategy(MyStrategy(), name="sub", num_cpus=1)

        assert isinstance(proxy, ActorProxy)
        assert strategy.strategies == {"sub": proxy}
        # the sub-strategy flag rides in the hydrate fields, applied when the actor is created
        create_actor.assert_not_called()
        assert proxy._hydrate_fields is not None
        assert proxy._hydrate_fields["is_substrategy"] is True


class TestAddComponentViaProxy:
    """The actor side of ActorProxy.add_*: the add method runs in the parent's process,
    a local child is not returned (it would be a copy), a remote child is."""

    @pytest.fixture
    def strategy(self, engine: TradeEngine) -> MyStrategy:
        strategy = MyStrategy()
        engine.add_strategy(strategy, "1m")
        return strategy

    def test_local_feature_is_added_and_none_is_returned(self, strategy: MyStrategy):
        feature = MyFeature()

        result = strategy._add_component_via_proxy(
            "add_feature", feature=feature, name="ema"
        )

        assert result is None
        assert strategy.features == {"ema": feature}
        assert feature.run_mode == RunMode.LOCAL

    def test_underlying_model_is_wrapped_added_and_none_is_returned(
        self, strategy: MyStrategy
    ):
        result = strategy._add_component_via_proxy(
            "add_model", model=LinearRegression()
        )

        assert result is None
        assert isinstance(strategy.models["LinearRegression"], SKLearnModel)

    def test_local_substrategy_is_added_and_none_is_returned(
        self, strategy: MyStrategy
    ):
        sub = MyStrategy()

        result = strategy._add_component_via_proxy(
            "add_strategy", strategy=sub, name="sub"
        )

        assert result is None
        assert strategy.strategies == {"sub": sub}
        assert sub.is_substrategy()

    def test_remote_child_is_returned(self, strategy: MyStrategy, create_actor: Any):
        result = strategy._add_component_via_proxy(
            "add_feature", feature=MyFeature(), name="ema", num_cpus=1
        )

        assert isinstance(result, ActorProxy)
        assert strategy.features == {"ema": result}
        create_actor.assert_not_called()


class TestAddThroughProxyParent:
    """ActorProxy.add_*: forwards to the parent's actor via _add_component_via_proxy."""

    @pytest.fixture
    def proxy(self, engine: TradeEngine) -> ActorProxy[MyStrategy]:
        return engine.add_strategy(MyStrategy(), "1m", name="alpha", num_cpus=1)

    @pytest.fixture
    def actor(self, create_actor: Any, ray_get: Any) -> Any:
        actor = create_actor.return_value
        actor._add_component_via_proxy.remote.return_value = None
        return actor

    def test_add_feature_is_forwarded_and_returns_none(
        self, proxy: ActorProxy[MyStrategy], actor: Any
    ):
        feature = MyFeature()

        added = proxy.add_feature(feature, "1m", name="ema", df_form="long")

        assert added is None
        actor._add_component_via_proxy.remote.assert_called_once_with(
            "add_feature",
            feature=feature,
            resolution="1m",
            name="ema",
            df_form="long",
            ray_actor_options=None,
        )

    def test_add_model_is_forwarded_unwrapped(
        self, proxy: ActorProxy[MyStrategy], actor: Any
    ):
        model = LinearRegression()

        added = proxy.add_model(model, name="lr")

        assert added is None
        # wrapping happens in the actor, in the parent's own add_model
        actor._add_component_via_proxy.remote.assert_called_once_with(
            "add_model",
            model=model,
            resolution="",
            name="lr",
            df_form="wide",
            ray_actor_options=None,
        )

    def test_add_strategy_is_forwarded(self, proxy: ActorProxy[MyStrategy], actor: Any):
        sub = MyStrategy()

        added = proxy.add_strategy(sub, name="sub")

        assert added is None
        actor._add_component_via_proxy.remote.assert_called_once_with(
            "add_strategy",
            strategy=sub,
            resolution="",
            name="sub",
            ray_actor_options=None,
        )

    def test_ray_kwargs_are_forwarded_and_the_child_proxy_is_returned(
        self, proxy: ActorProxy[MyStrategy], actor: Any, mocker: MockerFixture
    ):
        child_proxy = mocker.sentinel.child_proxy
        actor._add_component_via_proxy.remote.return_value = child_proxy
        feature = MyFeature()

        added = proxy.add_feature(feature, name="ema", num_cpus=1)

        assert added is child_proxy
        assert actor._add_component_via_proxy.remote.call_args.kwargs["num_cpus"] == 1

    def test_first_add_creates_and_hydrates_the_actor_once(
        self,
        engine: TradeEngine,
        proxy: ActorProxy[MyStrategy],
        create_actor: Any,
        actor: Any,
    ):
        create_actor.assert_not_called()

        proxy.add_feature(MyFeature(), name="a")
        proxy.add_feature(MyFeature(), name="b")

        create_actor.assert_called_once()
        actor._hydrate.remote.assert_called_once_with(
            name="alpha",
            run_mode=RunMode.REMOTE,
            resolution="1m",
            engine_context=engine.context,
            df_form="long",
        )
        assert actor._add_component_via_proxy.remote.call_count == 2


if TYPE_CHECKING:
    # NOTE: never run, ty checks that each add_* overload returns the documented type

    def _check_add_overloads(
        engine: TradeEngine,
        strategy: MyStrategy,
        strategy_proxy: ActorProxy[MyStrategy],
        feature: MyFeature,
        feature_proxy: ActorProxy[MyFeature],
        model: MyModel,
        model_proxy: ActorProxy[MyModel],
    ) -> None:
        # engine
        assert_type(engine.add_strategy(strategy, "1m"), MyStrategy)
        assert_type(
            engine.add_strategy(strategy, "1m", num_cpus=1), ActorProxy[MyStrategy]
        )

        # local parent
        assert_type(strategy.add_strategy(strategy), MyStrategy)
        assert_type(strategy.add_strategy(strategy, num_cpus=1), ActorProxy[MyStrategy])
        assert_type(strategy.add_strategy(strategy_proxy), ActorProxy[MyStrategy])
        assert_type(strategy.add_feature(feature), MyFeature)
        assert_type(strategy.add_feature(feature, num_cpus=1), ActorProxy[MyFeature])
        assert_type(strategy.add_feature(feature_proxy), ActorProxy[MyFeature])
        assert_type(strategy.add_model(model), MyModel)
        assert_type(strategy.add_model(model, num_cpus=1), ActorProxy[MyModel])
        assert_type(strategy.add_model(model_proxy), ActorProxy[MyModel])
        assert_type(strategy.add_model(LinearRegression()), BaseModel)
        assert_type(
            strategy.add_model(LinearRegression(), num_cpus=1), ActorProxy[BaseModel]
        )

        # remote parent (through its proxy)
        assert_type(strategy_proxy.add_strategy(strategy), None)
        assert_type(
            strategy_proxy.add_strategy(strategy, num_cpus=1), ActorProxy[MyStrategy]
        )
        assert_type(strategy_proxy.add_strategy(strategy_proxy), ActorProxy[MyStrategy])
        assert_type(strategy_proxy.add_feature(feature), None)
        assert_type(
            strategy_proxy.add_feature(feature, num_cpus=1), ActorProxy[MyFeature]
        )
        assert_type(strategy_proxy.add_feature(feature_proxy), ActorProxy[MyFeature])
        assert_type(strategy_proxy.add_model(model), None)
        assert_type(strategy_proxy.add_model(model, num_cpus=1), ActorProxy[MyModel])
        assert_type(strategy_proxy.add_model(model_proxy), ActorProxy[MyModel])
        assert_type(strategy_proxy.add_model(LinearRegression()), None)
        assert_type(
            strategy_proxy.add_model(LinearRegression(), num_cpus=1),
            ActorProxy[BaseModel],
        )
