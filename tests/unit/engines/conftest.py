from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from pytest_mock import MockerFixture

import pytest

from pfund.config import get_config
from pfund.engines.backtest_engine import BacktestEngine
from pfund.engines.trade_engine import TradeEngine
from pfund.utils.ray_dict import RayCompatibleDict


@pytest.fixture(autouse=True)
def isolated_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Points pfund's data/log/cache paths to a temp dir, engines write logs and lakehouse files there."""
    config = get_config()
    for attr in ("data_path", "log_path", "cache_path"):
        path = tmp_path / attr
        path.mkdir()
        monkeypatch.setattr(config, attr, path)


@pytest.fixture(autouse=True)
def reset_engine_singletons() -> Iterator[None]:
    yield
    TradeEngine._remove_singleton()
    BacktestEngine._remove_singleton()


@pytest.fixture(autouse=True)
def no_ray_shared_dicts(mocker: MockerFixture) -> None:
    """Creating an ActorProxy under a trade engine switches the zmq urls/ports dicts to Ray, which starts Ray."""
    mocker.patch.object(RayCompatibleDict, "enable_ray")
