from __future__ import annotations

from pfeed.sources.pfund.requests.component_base_request import (
    PFundComponentFeedBaseRequest,
)


class PFundComponentFeedDownloadRequest(PFundComponentFeedBaseRequest):
    extract_type: ExtractType = ExtractType.download
