"""Registry of trading venues, discovered via the "pfund.venues" entry points.

Every trading venue is a plugin package `pfund-xxx` (e.g. pfund-bybit), registering
an entry point named after the venue, pointing at its venue class:
    [project.entry-points."pfund.venues"]
    bybit = "pfund_bybit:Bybit"
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pfund.venues.venue_base import AnyVenue

from pfund_kit.plugins import PluginRegistry

ENTRY_POINT_GROUP = "pfund.venues"
plugin_registry = PluginRegistry(ENTRY_POINT_GROUP, kind="trading venue")
get_entry_points = plugin_registry.get_entry_points
list_venues = plugin_registry.list_names
get_entry_point = plugin_registry.get_entry_point


def get_venue(name: str) -> type[AnyVenue]:
    ep = get_entry_point(name)
    Venue: type[AnyVenue] = ep.load()
    if Venue.name != ep.name.upper():
        raise ValueError(f"entry point '{ep.name}' ({ep.value}) loads a venue whose name is {Venue.name}")
    return Venue
