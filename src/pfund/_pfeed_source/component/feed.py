# ty: ignore[incompatible-attribute-override,incompatible-mutable-override]
from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal, overload

if TYPE_CHECKING:
    from pfeed.io.base_io import BaseIO
    from pfeed.io.blob_io import BlobIO

    from pfund._pfeed_source.component.data_handler import (
        PFundComponentDataHandler,
        PFundComponentDataMetadata,
    )
    from pfund.typing import Component

import polars as pl
from pfeed.base.requests.base_request import BaseRequest
from pfeed.config import setup_logging
from pfund_blocks.enums import Environment

from pfund._pfeed_source.base.feed import PFundBaseFeed
from pfund._pfeed_source.component.data_model import PFundComponentDataModel
from pfund._pfeed_source.data_category import PFundDataCategory
from pfund.enums import ArtifactType


# TODO: add StreamingMixin
class PFundComponentFeed[PipelineT: bool = Literal[False]](
    PFundBaseFeed[
        BaseRequest,
        PFundComponentDataModel,
        PipelineT,
    ]
):
    DataModel: ClassVar[type[PFundComponentDataModel]] = PFundComponentDataModel
    data_domain: ClassVar[PFundDataCategory] = PFundDataCategory.COMPONENT_DATA

    @classmethod
    def create_data_model(
        cls,
        component: Component,
        artifact_type: ArtifactType | str,
        checkpoint_step: int | None = None,
        project: str | None = None,
        run: str | None = None,
    ) -> PFundComponentDataModel:
        """Create the data model of a component's artifact.

        Args:
            component: the component the artifact belongs to.
            artifact_type: the artifact, e.g. 'data', 'model'.
            checkpoint_step: step of the checkpoint artifact, only for artifact_type='checkpoint'.
        """
        artifact_type = ArtifactType(artifact_type.lower())
        return cls.DataModel.from_component(
            component=component,
            artifact_type=artifact_type,
            data_source=cls.DataSource.metadata.name,
            project=project,
            run=run,
            checkpoint_step=checkpoint_step,
        )

    def _validate_inputs(
        self,
        component: Component,
        artifact_type: ArtifactType,
        checkpoint_step: int | None,
    ):
        if (
            artifact_type in (ArtifactType.model, ArtifactType.checkpoint)
            and not component.is_model()
        ):
            raise ValueError(
                f"{component.name} must be a model for model/checkpoint artifacts"
            )
        if artifact_type == ArtifactType.checkpoint:
            if component.env != Environment.BACKTEST:
                raise ValueError(
                    "checkpoint artifacts are only available in backtesting"
                )
            if checkpoint_step is None:
                raise ValueError("checkpoint_step is required for a checkpoint")
        elif checkpoint_step is not None:
            raise ValueError("checkpoint_step is only valid for checkpoints")

    @staticmethod
    def _resolve_artifact_io(data_model: PFundComponentDataModel) -> BaseIO | BlobIO:
        component = data_model.component
        match data_model.artifact_type:
            case ArtifactType.data:
                # NOTE: the store's datalake, not settings.datalake, which is only the user's template (or None)
                return component.store.datalake
            case _:
                # the store builds the blob IO in the same lakehouse as the datalakes, so they share its storage
                return component.store.blob_io

    def load_metadata(
        self,
        component: Component,
        *,
        project: str | None = None,
        run: str | None = None,
    ) -> dict[str, Any] | None:
        """Read pfund's metadata.json of the component, None if it isn't stored.

        metadata.json describes the component (one per run per component), not an artifact,
        so it is read here instead of being returned by retrieve().
        Values come back as JSON types, e.g. dates as ISO strings.
        """
        # metadata.json's key doesn't depend on the artifact type, any artifact's handler locates it,
        # source is used since every component has one
        artifact_type = ArtifactType.source
        data_model = self.create_data_model(
            component, artifact_type, project=project, run=run
        )
        handler = data_model.DataHandler(
            data_model=data_model,
            io=self._resolve_artifact_io(data_model),
        )
        return handler._read_component_metadata()

    def save_artifact(
        self,
        component: Component,
        artifact_type: ArtifactType | str,
        *,
        artifact: pl.DataFrame | pl.LazyFrame | bytes,
        project: str | None = None,
        run: str | None = None,
        checkpoint_step: int | None = None,
    ) -> None:
        """Write a component artifact to storage, eagerly.

        The artifact is always passed in, it is never extracted from the component.

        The storage is decided by the artifact, see _resolve_artifact_io().
        Errors raise, nothing is recorded and swallowed like in a dataflow.

        Args:
            component: the component the artifact belongs to.
            artifact_type: The type of the artifact.
            artifact: The artifact to write, a polars DataFrame for the data artifact,
                bytes for the others.
            checkpoint_step: Step identifying the checkpoint artifact. Required
                when ``artifact_type`` is ``checkpoint`` and invalid otherwise.
        """
        setup_logging(env=component.env)
        artifact_type = ArtifactType(artifact_type.lower())
        self._validate_inputs(component, artifact_type, checkpoint_step)
        handler = self._create_data_handler(
            component,
            artifact_type,
            project=project,
            run=run,
            checkpoint_step=checkpoint_step,
        )
        # the handler writes a polars DataFrame or bytes
        handler.write_batch(
            artifact.collect() if isinstance(artifact, pl.LazyFrame) else artifact
        )

    @overload
    def load_artifact(
        self,
        component: Component,
        artifact_type: ArtifactType | str,
        *,
        project: str | None = ...,
        run: str | None = ...,
        checkpoint_step: int | None = ...,
        include_metadata: Literal[False] = ...,
    ) -> pl.LazyFrame | bytes | None: ...

    @overload
    def load_artifact(
        self,
        component: Component,
        artifact_type: ArtifactType | str,
        *,
        project: str | None = ...,
        run: str | None = ...,
        checkpoint_step: int | None = ...,
        include_metadata: Literal[True],
    ) -> tuple[pl.LazyFrame | bytes | None, PFundComponentDataMetadata | None]: ...

    def load_artifact(
        self,
        component: Component,
        artifact_type: ArtifactType | str,
        *,
        project: str | None = None,
        run: str | None = None,
        checkpoint_step: int | None = None,
        include_metadata: bool = False,
    ) -> (
        pl.LazyFrame
        | bytes
        | tuple[pl.LazyFrame | bytes | None, PFundComponentDataMetadata | None]
        | None
    ):
        """Read a component artifact back from the storage save_artifact() writes to, eagerly.

        Errors raise, so None only means the artifact isn't stored.
        pfund's metadata.json of the component is read by load_metadata().

        Args:
            include_metadata: also return the artifact's own metadata, e.g. its source_sha256 stamp.

        Returns:
            A LazyFrame for the data artifact, bytes for the others, None if it isn't stored.
            With include_metadata, (artifact, metadata), (None, None) if it isn't stored.
        """
        setup_logging(env=component.env)
        artifact_type = ArtifactType(artifact_type.lower())
        self._validate_inputs(component, artifact_type, checkpoint_step)
        handler = self._create_data_handler(
            component,
            artifact_type,
            project=project,
            run=run,
            checkpoint_step=checkpoint_step,
        )
        data, metadata = handler.read()
        if (
            artifact_type == ArtifactType.data
            and data is not None
            and not isinstance(data, pl.LazyFrame)
        ):
            raise TypeError(
                f"data artifact of '{component.name}' must be a polars LazyFrame, got {type(data).__name__}"
            )
        if not include_metadata:
            return data
        # an artifact is one partition: the data artifact is unpartitioned, a bytes artifact is read by its own partition
        return data, next(iter(metadata.values()), None)

    def _create_data_handler(
        self,
        component: Component,
        artifact_type: ArtifactType,
        *,
        project: str | None,
        run: str | None,
        checkpoint_step: int | None,
    ) -> PFundComponentDataHandler:
        data_model = self.create_data_model(
            component,
            artifact_type,
            checkpoint_step=checkpoint_step,
            project=project,
            run=run,
        )
        return data_model.DataHandler(
            data_model=data_model,
            io=self._resolve_artifact_io(data_model),
        )
