from pydantic import Field

from pfund.venues.venue_config import VenueConfig


class InteractiveBrokersConfig(VenueConfig):
    reqMktDepthL2: bool = Field(
        default=False,
        description="if true, reqMktDepthL2 is used over reqMktDepth for level-2 orderbook data",
    )
