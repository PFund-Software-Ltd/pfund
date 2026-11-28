from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal, assert_never, cast

if TYPE_CHECKING:
    from pfund.components.models.model_base import BaseModel

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
        return (
            self.run_path
            / self.component.component_type.to_plural()
            / self.component.name
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
                # the model decides its own serialization format, see BaseModel._artifact_extension
                extension = cast("BaseModel", component)._artifact_extension
                return ModelArtifact(extension=extension, **artifact_kwargs)
            case ArtifactType.data:
                return DataArtifact(**artifact_kwargs)
            case ArtifactType.source:
                return SourceArtifact(**artifact_kwargs)
            case ArtifactType.checkpoint:
                if checkpoint_step is None:
                    raise ValueError("checkpoint_step is required for a checkpoint")
                extension = cast("BaseModel", component)._checkpoint_extension
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


# NOTE: extension is a free str, the model sets it (BaseModel._artifact_extension / _checkpoint_extension),
# e.g. ".safetensors" (torch), ".joblib" (sklearn), ".bin" (default for a user model)
_EXTENSION_PATTERN = r"^\.[A-Za-z0-9]+$"


class DatasetArtifact(PFundComponentDataModel):
    """An artifact produced from the component's data selection: its trading df, model and checkpoints."""

    @property
    def data_sha256_path(self) -> Path:
        """The dir of the component's current data selection, e.g. .../strategies/SMACross/data=<data_sha256>.

        One dir per data selection (data, data range, dataset splits, split in use, see ArtifactMixin._data_fields()),
        so e.g. each cross-validation fold keeps its own trading df, model and checkpoints.
        """
        return self.component_path / f"data={self.component._data_sha256}"


class ModelArtifact(DatasetArtifact):
    extension: str = Field(pattern=_EXTENSION_PATTERN)


class CheckpointArtifact(DatasetArtifact):
    # the dir of the checkpoints in the data selection's dir, see PFundComponentDataHandler._create_dataset_key()
    CHECKPOINTS_DIR_NAME: ClassVar[str] = "checkpoints"

    extension: str = Field(pattern=_EXTENSION_PATTERN)
    step: int = Field(ge=0)


class DataArtifact(DatasetArtifact):
    """The output frame of a component (strategy / model / feature)."""

    TRADING_DF_DIR_NAME: ClassVar[str] = "trading_df"

    @property
    def artifact_dir(self) -> Path:
        return self.data_sha256_path / self.TRADING_DF_DIR_NAME
