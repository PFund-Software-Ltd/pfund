from pfund.venues.venue_base import BaseVenue
from pfund.venues.venue_metadata import VenueMetadata


class Alpaca(BaseVenue):
    METADATA = VenueMetadata()


async def place_orders(self):
    loop = asyncio.get_running_loop()
    resp = await loop.run_in_executor(
        None,  # None = default ThreadPoolExecutor
        lambda: requests.post(url, json=payload),  # a *sync* callable
    )
