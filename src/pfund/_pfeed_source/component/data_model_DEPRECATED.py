from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import Field, field_validator

from pfeed.base.data_model import BaseDataModel
from pfeed.sources.pfund.component_data_handler import PFundComponentDataHandler
from pfeed.sources.pfund.component_metadata import PFundComponentDataMetadata
from pfund.enums import ArtifactType, ComponentType
from pfund_blocks.enums.env import Environment


class PFundComponentDataModel(BaseDataModel):
    DataHandler: ClassVar[type[PFundComponentDataHandler]] = PFundComponentDataHandler

    artifact_type: ArtifactType
    extension: Literal[
        ".joblib",  # sklearn model
        ".safetensors",  # torch/jax model
        ".py",  # python source code
        ".delta",  # trading_df delta table
        ".pth",  # pytorch checkpoint
        ".pkl",  # jax checkpoint
    ]

    env: Environment | str
    project_name: str
    run_name: str
    component_type: ComponentType | str
    component_id: str
    metadata: PFundComponentDataMetadata | None = None

    @field_validator("artifact_type", mode="before")
    @classmethod
    def _validate_artifact_type(cls, v: str | ArtifactType) -> ArtifactType:
        if isinstance(v, str):
            return ArtifactType[v.lower()]
        return v

    @field_validator("env", mode="before")
    @classmethod
    def _validate_env(cls, v: str | Environment) -> Environment:
        if isinstance(v, str):
            return Environment(v.upper())
        return v

    @field_validator("component_type", mode="before")
    @classmethod
    def _validate_component_type(cls, v: str | ComponentType) -> ComponentType:
        if isinstance(v, str):
            return ComponentType[v.lower()]
        return v


class SourceArtifact(PFundComponentDataModel):
    artifact_type: Literal[ArtifactType.source] = ArtifactType.source
    extension: Literal[".py"] = ".py"
    filename: str  # e.g. "my_alpha.py"


class ModelArtifact(PFundComponentDataModel):
    artifact_type: Literal[ArtifactType.model] = ArtifactType.model
    extension: Literal[".joblib", ".safetensors"]


class CheckpointArtifact(PFundComponentDataModel):
    artifact_type: Literal[ArtifactType.checkpoint] = ArtifactType.checkpoint
    extension: Literal[".pth", ".pkl"]
    step: int = Field(ge=0)


class DataArtifact(PFundComponentDataModel):
    """The output frame of a component (strategy / model / feature)."""

    artifact_type: Literal[ArtifactType.data] = ArtifactType.data
    # .delta is just the convention for the deltalake folder
    extension: Literal[".delta"] = ".delta"
    # None appends. A predicate atomically replaces matching rows through
    # DeltaLakeIO's existing delete_where contract.
    replace_where: str | None = None
