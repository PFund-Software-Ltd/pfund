from __future__ import annotations

from typing import Any, ClassVar, Literal, assert_never

from pathlib import Path

from pydantic import Field, PrivateAttr

from pfund._pfeed_source.base.data_model import PFundBaseDataModel
from pfund._pfeed_source.component.data_handler import PFundComponentDataHandler
from pfund.enums import ArtifactType
from pfund.typing import Component


class PFundComponentDataModel(PFundBaseDataModel):
    DataHandler: ClassVar[type[PFundComponentDataHandler]] = PFundComponentDataHandler  # ty: ignore[incompatible-mutable-override]

    component: Component
    # set from the instance's type in model_post_init(), subclasses don't set it themselves
    _artifact_type: ArtifactType = PrivateAttr()

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        if isinstance(self, DataArtifact):
            self._artifact_type = ArtifactType.data
        elif isinstance(self, ModelArtifact):
            self._artifact_type = ArtifactType.model
        elif isinstance(self, SourceArtifact):
            self._artifact_type = ArtifactType.source
        elif isinstance(self, CheckpointArtifact):
            self._artifact_type = ArtifactType.checkpoint
        else:
            raise TypeError(f"unknown artifact data model {type(self).__name__}")

    @property
    def artifact_type(self) -> ArtifactType:
        return self._artifact_type

    @property
    def component_path(self) -> Path:
        """The component's dir, relative to the lakehouse path, e.g. runs/BACKTEST/default_project/default_run/strategies/SMACross-a1b2c3d4e5f6.

        The same as the namespace + name of the component's DatasetKey, see PFundComponentDataHandler._create_dataset_key().
        """
        return (
            self.run_path
            / self.component.component_type.to_plural()
            / self.component.component_id
        )

    @classmethod
    def from_component(
        cls,
        *,
        component: Component,
        artifact_type: ArtifactType,
        data_source: str,
        project: str | None = None,
        run: str | None = None,
        checkpoint_step: int | None = None,
    ) -> PFundComponentDataModel:
        """Creates the data model subclass of the artifact, e.g. ModelArtifact for artifact_type='model'."""
        artifact_kwargs: dict[str, Any] = {
            "data_source": data_source,
            "env": component.env,
            "component": component,
        }
        if project is not None:
            artifact_kwargs["project"] = project
        if run is not None:
            artifact_kwargs["run"] = run
        match artifact_type:
            case ArtifactType.model:
                from pfund.components.models.pytorch_model import PyTorchModel
                from pfund.components.models.sklearn_model import SKLearnModel

                if isinstance(component, SKLearnModel):
                    extension = ".joblib"
                elif isinstance(component, PyTorchModel):
                    extension = ".safetensors"
                else:
                    raise ValueError(f"Unsupported model type: {type(component)}")
                return ModelArtifact(extension=extension, **artifact_kwargs)
            case ArtifactType.data:
                return DataArtifact(**artifact_kwargs)
            case ArtifactType.source:
                return SourceArtifact(
                    filename=component._source_file_path.name, **artifact_kwargs
                )
            case ArtifactType.checkpoint:
                from pfund.components.models.pytorch_model import PyTorchModel

                if checkpoint_step is None:
                    raise ValueError("checkpoint_step is required for a checkpoint")
                if isinstance(component, PyTorchModel):
                    extension = ".pth"
                else:
                    raise ValueError(
                        f"Unsupported checkpoint model type: {type(component)}"
                    )
                return CheckpointArtifact(
                    extension=extension, step=checkpoint_step, **artifact_kwargs
                )
            case _:
                assert_never(artifact_type)

    def __str__(self) -> str:
        return ":".join(
            [
                super().__str__(),
                self.component.name,
                self.artifact_type,
            ]
        )


# NOTE: the extension is the format pfund serialized the bytes in, data artifacts are frames,
# their storage format is decided by the IO, so they have no extension
class SourceArtifact(PFundComponentDataModel):
    extension: Literal[".py"] = ".py"  # python source code
    filename: str  # e.g. "my_alpha.py"


class ModelArtifact(PFundComponentDataModel):
    extension: Literal[
        ".joblib",  # sklearn model
        ".safetensors",  # torch/jax model
    ]


class CheckpointArtifact(PFundComponentDataModel):
    extension: Literal[
        ".pth",  # pytorch checkpoint
        ".pkl",  # jax checkpoint
    ]
    step: int = Field(ge=0)


class DataArtifact(PFundComponentDataModel):
    """The output frame of a component (strategy / model / feature)."""
