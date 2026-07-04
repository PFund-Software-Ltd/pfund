from typing import Literal

from pydantic import BaseModel, Field, ConfigDict


class VenueConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    cancel_all_at: list[Literal["start", "reconnect", "stop"]] = Field(
        default=["start", "reconnect", "stop"],
    )
    refetch_markets: bool = Field(
        default=False,
        description="refetch markets.yml on startup, if the trading venue has an endpoint to fetch all trading markets.",
    )
    stream_api_first: bool = Field(
        default=True,
        description="Always use the Stream API (e.g. WebSocket API) for actions such as placing or canceling orders, even if RESTful API is available.",
    )
