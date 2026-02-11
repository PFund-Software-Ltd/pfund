from typing import Any

from pydantic import BaseModel, Field, ConfigDict

from pfeed.enums import IOFormat
from pfund.enums.database import Database


class BaseEngineSettings(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid', frozen=True)

    storage_options: dict[Database, dict[str, Any]] = Field(default_factory=dict)
    io_options: dict[IOFormat, dict[str, Any]] = Field(default_factory=dict)
 