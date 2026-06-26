from typing import Literal

from pydantic import BaseModel, Field, ConfigDict

from pfund.enums import DataChannel


class VenueConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    cancel_all_at: list[Literal["start", "reconnect", "stop"]] = Field(
        default=["start", "reconnect", "stop"],
    )
    refetch_markets: bool = Field(
        default=False,
        description="refetch markets.yml on startup.",
    )
    stream_api_first: bool = Field(
        default=True,
        description="Always use the Stream API (e.g. WebSocket API) for actions such as placing or canceling orders, even if RESTful API is available.",
    )
    private_channels: list[
        Literal[
            DataChannel.balance,
            DataChannel.position,
            DataChannel.order,
            DataChannel.trade,
        ]
        | Literal["balance", "position", "order", "trade"]
    ] = Field(
        default=[
            DataChannel.balance,
            DataChannel.position,
            DataChannel.order,
            DataChannel.trade,
        ],
        description="private channels to subscribe to.",
    )
