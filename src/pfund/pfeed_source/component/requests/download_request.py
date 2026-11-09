from __future__ import annotations

from typing import ClassVar

from pfeed_pfund.component.data_handler import (
    ComponentMetadata,
    PFundComponentDataMetadata,
    RunMetadata,
)
from pfeed_pfund.component.requests.base_request import PFundComponentFeedBaseRequest


class PFundComponentFeedDownloadRequest(PFundComponentFeedBaseRequest):
    extract_type: ClassVar[str] = "download"

    def _create_metadata(self) -> PFundComponentDataMetadata:
        component = self.component
        component_snapshot = component.to_dict()
        component_type = component_snapshot.pop("component_type")
        run_mode = component_snapshot.pop("run_mode")
        data_start = component_snapshot.pop("data_start")
        data_end = component_snapshot.pop("data_end")
        settings = component_snapshot.pop("settings")
        return PFundComponentDataMetadata(
            component_id=component.component_id,
            component_type=str(component_type),
            component=ComponentMetadata(**component_snapshot),
            run=RunMetadata(
                env=str(self.env),
                project_name=self.project_name,
                run_name=self.run_name,
                run_mode=str(run_mode),
                data_start=data_start,
                data_end=data_end,
                settings=settings,
            ),
        )
