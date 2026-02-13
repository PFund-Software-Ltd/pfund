from typing import Any

import os

from pydantic import BaseModel, Field, ConfigDict

from pfeed.enums import IOFormat
from pfund.enums.database import Database


class BaseEngineSettings(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid', frozen=True)

    auto_download_data: bool = Field(
        default=False,
        description="""
            if True, the engine will automatically download data using pfeed if no data is found in the storage specified in add_data()'s storage_config.
            if False, the engine will not download data automatically.
        """,
    )
    num_batch_workers: int | None = Field(
        default=os.cpu_count(),
        description="""
            number of workers to fetch data using pfeed, equivalent to the parameter 'num_batch_workers' in pfeed.
            if None, Ray will NOT be used for fetching data and it will be done sequentially.
        """,
    )
    storage_options: dict[Database, dict[str, Any]] = Field(default_factory=dict)
    io_options: dict[IOFormat, dict[str, Any]] = Field(default_factory=dict)
 