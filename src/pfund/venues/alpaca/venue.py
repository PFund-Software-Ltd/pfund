from pfund.venues.venue_base import BaseVenue
from pfund.venues.venue_metadata import VenueMetadata


class Alpaca(BaseVenue):
    METADATA = VenueMetadata(
        requires_asyncio_loop=True,
    )


async def place_orders(self):
    resp = await self.loop.run_in_executor(
        None,  # None = default ThreadPoolExecutor
        lambda: requests.post(url, json=payload),  # a *sync* callable
    )
