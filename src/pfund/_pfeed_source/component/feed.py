# ty: ignore[incompatible-attribute-override,incompatible-mutable-override]
from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal, assert_never, cast

if TYPE_CHECKING:
    from pfeed.io.base_io import BaseIO
    from pfeed.io.blob_io import BlobIO

    from pfund._pfeed_source.component.data_handler import PFundComponentDataHandler
    from pfund.components.models.model_base import BaseModel
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
    def _resolve_artifact_io(
        component: Component, artifact_type: ArtifactType
    ) -> BaseIO | BlobIO:
        match artifact_type:
            case ArtifactType.data:
                # NOTE: store.datalake, not settings.datalake, which is only the user's template
                # (or None), the store rebuilds it under the engine's scope in setup_datalake()
                return component.store.datalake
            case _:
                from pfeed.io.blob_io import BlobIO

                # BlobIO's own default is under pfeed's data_path, so pass pfund's (engine-scoped) one
                return BlobIO(
                    base_path=str(
                        component.context.config.data_path / BlobIO.DEFAULT_DIR_NAME
                    )
                )

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
            io=self._resolve_artifact_io(component, artifact_type),
        )
        return handler._read_component_metadata()

    def save_artifact(
        self,
        component: Component,
        artifact_type: ArtifactType | str,
        *,
        project: str | None = None,
        run: str | None = None,
        checkpoint_step: int | None = None,
    ) -> None:
        """Extract a component artifact and write it to storage, eagerly.

        The storage is decided by the artifact, see _resolve_artifact_io().
        Errors raise, nothing is recorded and swallowed like in a dataflow.

        Args:
            component: the component to extract the artifact from.
            artifact_type: The artifact to extract.
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
        data = self._extract_artifact(component, artifact_type)
        # the handler writes a polars DataFrame or bytes
        handler.write_batch(data.collect() if isinstance(data, pl.LazyFrame) else data)

    def load_artifact(
        self,
        component: Component,
        artifact_type: ArtifactType | str,
        *,
        project: str | None = None,
        run: str | None = None,
        checkpoint_step: int | None = None,
    ) -> pl.LazyFrame | bytes | None:
        """Read a component artifact back from the storage save_artifact() writes to, eagerly.

        Errors raise, so None only means the artifact isn't stored.
        pfund's metadata.json of the component is read by retrieve_metadata().

        Returns:
            A LazyFrame for the data artifact, bytes for the others, None if it isn't stored.
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
        data, _ = handler.read()
        return data

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
            io=self._resolve_artifact_io(component, artifact_type),
        )

    @staticmethod
    def _extract_artifact(
        component: Component, artifact_type: ArtifactType
    ) -> pl.LazyFrame | bytes:
        match artifact_type:
            case ArtifactType.source:
                return component._source_artifact.read_bytes()
            case ArtifactType.model:
                return cast("BaseModel", component)._model_artifact
            case ArtifactType.data:
                from pfeed.utils.dataframe import convert_dataframe

                # the component's df can be of any data tool, pfeed handles polars only
                return cast("pl.LazyFrame", convert_dataframe(component._data_artifact))
            case ArtifactType.checkpoint:
                checkpoint_artifact = cast("BaseModel", component)._checkpoint_artifact
                if checkpoint_artifact is None:
                    raise RuntimeError("No checkpoint is staged for download")
                return checkpoint_artifact
            case _:
                assert_never(artifact_type)
