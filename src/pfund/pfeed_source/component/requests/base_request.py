from __future__ import annotations

from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    from pfeed_pfund.component.data_handler import PFundComponentDataMetadata
    from pfeed_pfund.component.data_model import PFundComponentDataModel

from pfund.components.mixin import ComponentMixin
from pfund.enums import ArtifactType
from pfund_blocks.enums.env import Environment
from pydantic import Field, field_validator, model_validator

from pfeed_pfund.base_request import PFundBaseRequest


class PFundComponentFeedBaseRequest(PFundBaseRequest):
    # the live component, the artifact is extracted from it (download), or it identifies the stored artifact (retrieve)
    component: ComponentMixin
    artifact_type: ArtifactType
    checkpoint_step: int | None = Field(
        default=None, description="model checkpoint step", ge=0
    )

    @field_validator("artifact_type", mode="before")
    @classmethod
    def _validate_artifact_type(cls, value: ArtifactType | str) -> ArtifactType:
        if isinstance(value, str):
            return ArtifactType[value.lower()]
        return value

    @model_validator(mode="after")
    def _check_artifact_type(self) -> Self:
        if (
            self.artifact_type in (ArtifactType.model, ArtifactType.checkpoint)
            and not self.component.is_model()
        ):
            raise ValueError(
                f"{self.component.name} must be a model for model/checkpoint artifacts"
            )
        if self.artifact_type == ArtifactType.checkpoint:
            if self.env != Environment.BACKTEST:
                raise ValueError("checkpoint artifacts are only available in backtesting")
            if self.checkpoint_step is None:
                raise ValueError("checkpoint_step is required for a checkpoint")
        elif self.checkpoint_step is not None:
            raise ValueError("checkpoint_step is only valid for checkpoints")
        return self

    def _create_metadata(self) -> PFundComponentDataMetadata | None:
        """The component snapshot stored with the artifact, None if the request doesn't write it."""
        return None

    def to_data_model(self) -> PFundComponentDataModel:
        from pfeed_pfund.component.data_model import PFundComponentDataModel

        return PFundComponentDataModel.for_artifact(
            self.component,
            self.artifact_type,
            data_source=self.data_source.name,
            data_origin=self.data_origin,
            env=self.env,
            project_name=self.project_name,
            run_name=self.run_name,
            checkpoint_step=self.checkpoint_step,
            metadata=self._create_metadata(),
        )

    def __str__(self) -> str:
        from pprint import pformat

        data: dict[str, str] = {
            "data_source": self.data_source.name,
            "env": str(self.env),
            "project_name": self.project_name,
            "run_name": self.run_name,
            "component": self.component.name,
            "artifact_type": str(self.artifact_type),
        }
        if self.checkpoint_step is not None:
            data["checkpoint_step"] = str(self.checkpoint_step)
        if self.io:
            data["io"] = repr(self.io)
        return pformat(data, sort_dicts=False)
