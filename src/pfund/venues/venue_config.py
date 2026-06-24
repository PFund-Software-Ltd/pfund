from typing import Literal

from pydantic import BaseModel, Field

from pfund.enums import DataChannel


class VenueConfig(BaseModel):
    cancel_all_at_start: bool = False
    cancel_all_at_reconnection: bool = False
    cancel_all_at_stop: bool = False
    # force refetching market configs
    refetch_markets: bool = Field(default=False)
    # renew market configs every x days
    renew_markets_every_x_days: int = Field(default=7)
    # Always use the Stream API (e.g. WebSocket API) for actions like placing or canceling orders, even if RESTful API is available.
    stream_api_first: bool = Field(default=True)
    subscribed_private_channels: list[
        Literal[
            DataChannel.balance,
            DataChannel.position,
            DataChannel.order,
            DataChannel.trade,
        ]
    ] = Field(
        default=[
            DataChannel.balance,
            DataChannel.position,
            DataChannel.order,
            DataChannel.trade,
        ]
    )
