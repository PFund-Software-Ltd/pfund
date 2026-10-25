"""Tests pfund.registry against fake "pfund.venues" entry points, so no real plugin has to be installed."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from pytest_mock import MockerFixture

import sys
import types
from importlib.metadata import EntryPoint

import pytest
from pfund_kit import plugins

from pfund import registry

FAKE_MODULE = "fake_pfund_plugin"


class FakeVenue:
    name = "FAKE"


class MismatchedVenue:
    # entry point is named "fake" but the venue's name says otherwise
    name = "OTHER"


def _ep(name: str, attr: str = "FakeVenue") -> EntryPoint:
    return EntryPoint(name=name, value=f"{FAKE_MODULE}:{attr}", group=registry.ENTRY_POINT_GROUP)


@pytest.fixture
def install_plugins(mocker: MockerFixture) -> Iterator[Callable[..., None]]:
    """Replaces the installed entry points with the given fake ones."""
    module = types.ModuleType(FAKE_MODULE)
    for cls in (FakeVenue, MismatchedVenue):
        setattr(module, cls.__name__, cls)
    mocker.patch.dict(sys.modules, {FAKE_MODULE: module})

    def _install(*eps: EntryPoint) -> None:
        mocker.patch.object(plugins, "entry_points", lambda group: [ep for ep in eps if ep.group == group])
        registry.plugin_registry.cache_clear()

    yield _install
    registry.plugin_registry.cache_clear()


def test_list_venues(install_plugins: Callable[..., None]):
    install_plugins(_ep("zeta"), _ep("fake"))
    assert registry.list_venues() == ["FAKE", "ZETA"]


def test_get_venue_unknown(install_plugins: Callable[..., None]):
    install_plugins(_ep("fake"))
    with pytest.raises(ValueError, match=r"unknown trading venue 'nope', installed trading venues: \['FAKE'\]"):
        registry.get_venue("nope")


def test_get_venue(install_plugins: Callable[..., None]):
    install_plugins(_ep("fake"))
    assert registry.get_venue("Fake") is FakeVenue


def test_get_venue_rejects_mismatched_venue_name(install_plugins: Callable[..., None]):
    install_plugins(_ep("fake", attr="MismatchedVenue"))
    with pytest.raises(ValueError, match="venue whose name is OTHER"):
        registry.get_venue("fake")
