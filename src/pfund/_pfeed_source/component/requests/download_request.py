from __future__ import annotations

from typing import ClassVar

from pfund._pfeed_source.component.requests.base_request import (
    PFundComponentFeedBaseRequest,
)


class PFundComponentFeedDownloadRequest(PFundComponentFeedBaseRequest):
    extract_type: ClassVar[str] = "download"
