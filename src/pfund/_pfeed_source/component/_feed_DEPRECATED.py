# ty: ignore[incompatible-attribute-override,incompatible-mutable-override]
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
    Self,
    assert_never,
    cast,
    override,
)

if TYPE_CHECKING:
    from pfeed.dataflow.result import RunResult
    from pfeed.io.base_io import BaseIO
    from pfeed.io.blob_io import BlobIO

    from pfund._pfeed_source.component.requests_DEPRECATED.base_request import (
        PFundComponentFeedBaseRequest,
    )
    from pfund.components.models.model_base import BaseModel
    from pfund.typing import Component

import polars as pl
from pfeed.base.feed import requires
from pfeed.config import setup_logging
from pfund_blocks.enums import Environment

from pfund._pfeed_source.base.feed import PFundBaseFeed
from pfund._pfeed_source.component.data_model import PFundComponentDataModel
from pfund._pfeed_source.component.requests_DEPRECATED import (
    PFundComponentFeedDownloadRequest,
    PFundComponentFeedRetrieveRequest,
)
from pfund._pfeed_source.data_category import PFundDataCategory
from pfund.enums import ArtifactType


# TODO: add StreamingMixin
class PFundComponentFeed[PipelineT: bool = Literal[False]](
    PFundBaseFeed[
        "PFundComponentFeedBaseRequest",
        PFundComponentDataModel,
        PipelineT,
    ]
):
    """Feed over a pfund component's artifacts (data, model, source).

    Architecture (see the design discussion):
      - `download(artifact_type)` is the EXTRACT: it pulls the artifact out of the
        component.
      - `load(...)` is the LOAD: it reuses BaseFeed's storage machinery to write the
        downloaded artifact. The IO format is chosen from the artifact's extension —
        `.delta` -> DeltaLakeIO (frames, pfeed's native currency), everything else ->
        FileIO via IOFormat.BLOB (opaque bytes; pfund owns the serialization).
      - `retrieve(...)` is the read-back twin of `download()`+`load()`: same identity,
        but the extract source is pfeed's STORAGE instead of the live component. No
        serialization happens — FileIO/DeltaLakeIO hand the persisted bytes/frame back.
    """

    class Requests:
        """Request class per verb, None = not supported."""

        download: ClassVar[type[PFundComponentFeedDownloadRequest] | None] = (
            PFundComponentFeedDownloadRequest
        )
        retrieve: ClassVar[type[PFundComponentFeedRetrieveRequest] | None] = (
            PFundComponentFeedRetrieveRequest
        )
        # TODO: connect to mtflow's ws server
        # streaming a component's live signals — separate (StreamingFeedMixin) path
        stream: ClassVar[None] = None

    DataModel: ClassVar[type[PFundComponentDataModel]] = PFundComponentDataModel
    data_domain: ClassVar[PFundDataCategory] = PFundDataCategory.COMPONENT_DATA
    _required_methods: ClassVar[dict[str, tuple[str, ...]]] = {
        PFundComponentFeedDownloadRequest.extract_type: ("_download_impl",),
        PFundComponentFeedRetrieveRequest.extract_type: ("_retrieve_impl",),
    }

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
                # NOTE: context.datalake, not settings.datalake, which is None unless the user set one
                return component.context.datalake
            case _:
                from pfeed.io.blob_io import BlobIO

                # BlobIO's own default is under pfeed's data_path, so pass pfund's (engine-scoped) one
                return BlobIO(
                    base_path=str(
                        component.context.config.data_path / BlobIO.DEFAULT_DIR_NAME
                    )
                )

    @requires("download")
    def download(
        self,
        component: Component,
        artifact_type: ArtifactType | str,
        *,
        project: str | None = None,
        run: str | None = None,
        checkpoint_step: int | None = None,
    ) -> Self | RunResult[pl.LazyFrame | bytes]:
        """Extract a component artifact and persist it to storage.

        The storage is decided by the artifact, not the caller: the data artifact (a frame) goes to
        the engine's datalake, the others (bytes) to a BlobIO under pfund's data_path.

        Args:
            component: the component to extract the artifact from.
            artifact_type: The artifact to extract.
            checkpoint_step: Step identifying the checkpoint artifact. Required
                when ``artifact_type`` is ``checkpoint`` and invalid otherwise.
        Returns:
            The feed in pipeline mode; otherwise the completed run result.
        """
        env = component.env
        setup_logging(env=env)
        artifact_type = ArtifactType(artifact_type.lower())
        self._validate_inputs(component, artifact_type, checkpoint_step)
        io = self._resolve_artifact_io(component, artifact_type)
        request = cast(
            "type[PFundComponentFeedDownloadRequest]", self.Requests.download
        )(
            data_source=self.data_source,
            env=env,
            component=component,
            artifact_type=artifact_type,
            checkpoint_step=checkpoint_step,
            project=project,
            run=run,
            io=io,
        )
        self._append_request(request)
        self._create_batch_dataflows(
            extract_func=self._download_impl,
            data_models=[request.to_data_model()],
        )
        return self.run() if not self.is_pipeline() else self

    @staticmethod
    def _download_impl(
        request: PFundComponentFeedBaseRequest,
        data_model: PFundComponentDataModel,
    ) -> pl.LazyFrame | bytes:
        component = request.component
        match data_model.artifact_type:
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
                assert_never(data_model.artifact_type)

    @requires("retrieve")
    def retrieve(
        self,
        component: Component,
        artifact_type: ArtifactType | str,
        *,
        project: str | None = None,
        run: str | None = None,
        checkpoint_step: int | None = None,
    ) -> Self | RunResult[pl.LazyFrame | bytes]:
        env = component.env
        setup_logging(env=env)
        artifact_type = ArtifactType(artifact_type.lower())
        self._validate_inputs(component, artifact_type, checkpoint_step)
        io = self._resolve_artifact_io(component, artifact_type)
        request = cast(
            "type[PFundComponentFeedRetrieveRequest]", self.Requests.retrieve
        )(
            data_source=self.data_source,
            io_for_retrieval=io,
            component=component,
            env=env,
            project=project,
            run=run,
            artifact_type=artifact_type,
            checkpoint_step=checkpoint_step,
        )
        self._append_request(request)
        self._create_batch_dataflows(
            extract_func=self._retrieve_impl,
            data_models=[request.to_data_model()],
        )
        return self.run() if not self.is_pipeline() else self

    @staticmethod
    def _retrieve_impl(
        request: PFundComponentFeedBaseRequest, data_model: PFundComponentDataModel
    ) -> pl.LazyFrame | bytes | None:
        assert isinstance(request, PFundComponentFeedRetrieveRequest)
        # like pfeed's feeds, the result only needs the data, read() also returns the metadata
        data, _ = request.create_handler(data_model).read()
        return data

    def retrieve_metadata(
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

    @override
    def _append_request(self, request: PFundComponentFeedBaseRequest) -> None:
        # one artifact per run: the artifacts can't be combined into one result (a frame and bytes),
        # so pipeline mode can't queue a second request before run()
        if self._requests:
            raise ValueError(
                f"{self.name}'s component feed runs one request at a time, call run() before the next "
                + "download()/retrieve()"
            )
        super()._append_request(request)

    @override
    def run(
        self, use_prefect: bool = False, **prefect_kwargs: Any
    ) -> RunResult[pl.LazyFrame | bytes]:
        """Run the request's dataflow.

        A failed dataflow doesn't raise, its error is recorded in the result, like pfeed's feeds.

        Args:
            use_prefect: If True, run the dataflow as a Prefect flow.
            prefect_kwargs: kwargs for Prefect's `@flow` decorator, e.g. `retries=2`.

        Returns:
            RunResult with the artifact, a LazyFrame for data, bytes for the others,
            None if there is none (e.g. not stored, or the dataflow failed).
        """
        from pfeed.dataflow.result import RunResult

        dataflows = self._run_batch_dataflows(
            use_prefect=use_prefect, prefect_kwargs=prefect_kwargs
        )
        # one component, one artifact type, one request (enforced in _append_request),
        # so there is exactly one dataflow — nothing to aggregate.
        [dataflow] = dataflows
        # _download_impl/_retrieve_impl only produce a polars LazyFrame or bytes, no other dataframe
        data = cast("pl.LazyFrame | bytes | None", dataflow.result.data)
        return RunResult(data=data, dataflows=dataflows)
