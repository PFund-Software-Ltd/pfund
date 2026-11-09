from __future__ import annotations

from typing import Any

from pydantic import Field

from pfeed.io.io_config import IOConfig
from pfeed.sources.pfund.requests.component_base_request import (
    PFundComponentFeedBaseRequest,
)
from pfeed.storages.storage_config import StorageConfig


class PFundComponentFeedRetrieveRequest(PFundComponentFeedBaseRequest):
    extract_type: ExtractType = ExtractType.retrieve
    storage_config_for_retrieval: StorageConfig = Field(
        description="Storage configuration used for retrieval, not for loading data to storage"
    )
    io_config_for_retrieval: IOConfig = Field(
        description="IO configuration used for data retrieval, not for loading data to storage"
    )

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        if not self.storage_config_for_retrieval:
            raise ValueError("storage config is missing, cannot retrieve artifact")
