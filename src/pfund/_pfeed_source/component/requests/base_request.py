from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pfund._pfeed_source.component.data_model import PFundComponentDataModel

from pydantic import Field

from pfund._pfeed_source.base.requests.base_request import PFundBaseRequest
from pfund.enums import ArtifactType
from pfund.typing import Component


class PFundComponentFeedBaseRequest(PFundBaseRequest):
    # name of the component the artifact belongs to, the stored artifact is identified by its component_id
    component: Component
    artifact_type: ArtifactType
    checkpoint_step: int | None = Field(
        default=None, description="model checkpoint step", ge=0
    )

    def to_data_model(self) -> PFundComponentDataModel:
        from pfund._pfeed_source.component.data_model import PFundComponentDataModel

        return PFundComponentDataModel.from_component(
            component=self.component,
            artifact_type=self.artifact_type,
            data_source=self.data_source.name,
            project=self.project,
            run=self.run,
            checkpoint_step=self.checkpoint_step,
        )

    def __str__(self) -> str:
        from pprint import pformat

        data: dict[str, str] = {
            "data_source": self.data_source.name,
            "env": str(self.env),
            "artifact_type": str(self.artifact_type),
            "component_name": self.component.name,
        }
        if self.checkpoint_step is not None:
            data["checkpoint_step"] = str(self.checkpoint_step)
        if self.io:
            data["io"] = repr(self.io)
        return pformat(data, sort_dicts=False)
