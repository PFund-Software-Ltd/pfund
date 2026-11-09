from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal, assert_never

if TYPE_CHECKING:
    from pfund.typing import Component

from pfeed.base.data_model import BaseDataModel
from pfund.enums import ArtifactType, ComponentType
from pfund_blocks.enums.env import Environment
from pydantic import Field, field_validator

from pfeed_pfund.component.data_handler import (
    PFundComponentDataHandler,
    PFundComponentDataMetadata,
)


class PFundComponentDataModel(BaseDataModel):
    DataHandler: ClassVar[type[PFundComponentDataHandler]] = PFundComponentDataHandler

    artifact_type: ArtifactType
    env: Environment
    project_name: str
    run_name: str
    component_type: ComponentType
    component_id: str
    # the component snapshot stored with the artifact, set when downloading, None when retrieving
    metadata: PFundComponentDataMetadata | None = None

    @classmethod
    def for_artifact(
        cls,
        component: Component,
        artifact_type: ArtifactType | str,
        *,
        data_source: str,
        env: Environment | str,
        project_name: str,
        run_name: str,
        data_origin: str = "",
        checkpoint_step: int | None = None,
        metadata: PFundComponentDataMetadata | None = None,
    ) -> PFundComponentDataModel:
        """Creates the data model subclass of the artifact, e.g. ModelArtifact for artifact_type='model'."""
        artifact_type = ArtifactType[artifact_type.lower()]
        artifact_kwargs: dict[str, Any] = {
            "data_source": data_source,
            "data_origin": data_origin,
            "env": env,
            "project_name": project_name,
            "run_name": run_name,
            "component_type": component.component_type,
            "component_id": component.component_id,
            "metadata": metadata,
        }
        match artifact_type:
            case ArtifactType.model:
                from pfund.components.models.jax_model import JAXModel
                from pfund.components.models.pytorch_model import PyTorchModel
                from pfund.components.models.sklearn_model import SKLearnModel

                if isinstance(component, SKLearnModel):
                    extension = ".joblib"
                elif isinstance(component, (PyTorchModel, JAXModel)):
                    extension = ".safetensors"
                else:
                    raise ValueError(f"Unsupported model type: {type(component)}")
                return ModelArtifact(extension=extension, **artifact_kwargs)
            case ArtifactType.data:
                return DataArtifact(**artifact_kwargs)
            case ArtifactType.source:
                return SourceArtifact(filename=component._source_artifact.name, **artifact_kwargs)
            case ArtifactType.checkpoint:
                from pfund.components.models.jax_model import JAXModel
                from pfund.components.models.pytorch_model import PyTorchModel

                if checkpoint_step is None:
                    raise ValueError("checkpoint_step is required for a checkpoint")
                if isinstance(component, PyTorchModel):
                    extension = ".pth"
                elif isinstance(component, JAXModel):
                    extension = ".pkl"
                else:
                    raise ValueError(f"Unsupported checkpoint model type: {type(component)}")
                return CheckpointArtifact(extension=extension, step=checkpoint_step, **artifact_kwargs)
            case _:
                assert_never(artifact_type)

    def __str__(self) -> str:
        return ":".join(
            [
                self.env,
                super().__str__(),
                self.project_name,
                self.run_name,
                self.component_type,
                self.component_id,
                self.artifact_type,
            ]
        )

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

    # also lowered by PFundBaseRequest, here too since create_data_model() doesn't go through a request
    @field_validator("project_name", "run_name", mode="before")
    @classmethod
    def _lower_names(cls, value: str) -> str:
        return value.lower()

    @field_validator("component_type", mode="before")
    @classmethod
    def _validate_component_type(cls, v: str | ComponentType) -> ComponentType:
        if isinstance(v, str):
            return ComponentType[v.lower()]
        return v


# NOTE: the extension is the format pfund serialized the bytes in, data artifacts are frames,
# their storage format is decided by the IO, so they have no extension
class SourceArtifact(PFundComponentDataModel):
    artifact_type: Literal[ArtifactType.source] = ArtifactType.source
    extension: Literal[".py"] = ".py"  # python source code
    filename: str  # e.g. "my_alpha.py"


class ModelArtifact(PFundComponentDataModel):
    artifact_type: Literal[ArtifactType.model] = ArtifactType.model
    extension: Literal[
        ".joblib",  # sklearn model
        ".safetensors",  # torch/jax model
    ]


class CheckpointArtifact(PFundComponentDataModel):
    artifact_type: Literal[ArtifactType.checkpoint] = ArtifactType.checkpoint
    extension: Literal[
        ".pth",  # pytorch checkpoint
        ".pkl",  # jax checkpoint
    ]
    step: int = Field(ge=0)


class DataArtifact(PFundComponentDataModel):
    """The output frame of a component (strategy / model / feature)."""

    artifact_type: Literal[ArtifactType.data] = ArtifactType.data
