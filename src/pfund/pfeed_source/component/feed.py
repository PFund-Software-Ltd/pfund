from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Self, assert_never, cast

if TYPE_CHECKING:
    from collections.abc import Callable

    from pfeed.dataflow.result import RunResult
    from pfeed.io.base_io import BaseIO
    from pfund.components.models.model_base import BaseModel
    from pfund.typing import Component

    from pfeed_pfund.component.requests.base_request import (
        PFundComponentFeedBaseRequest,
    )
    from pfeed_pfund.source import PFund  # noqa: F401, used as a string forward reference in the class bases

import polars as pl
from pfeed.base.feed import BaseFeed, requires
from pfeed.config import setup_logging
from pfeed.enums import DataCategory
from pfund.enums import ArtifactType

from pfeed_pfund.component.data_model import PFundComponentDataModel
from pfeed_pfund.component.requests import (
    PFundComponentFeedDownloadRequest,
    PFundComponentFeedRetrieveRequest,
)


class PFundComponentFeed(
    BaseFeed["PFund", "PFundComponentFeedBaseRequest", PFundComponentDataModel]
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

    class Requests:  # ty: ignore[invalid-attribute-override]
        """Request class per verb, None = not supported."""

        download: ClassVar[type[PFundComponentFeedDownloadRequest] | None] = PFundComponentFeedDownloadRequest
        retrieve: ClassVar[type[PFundComponentFeedRetrieveRequest] | None] = PFundComponentFeedRetrieveRequest
        # TODO: connect to mtflow's ws server
        # streaming a component's live signals — separate (StreamingFeedMixin) path
        stream: ClassVar[None] = None

    DataModel: ClassVar[type[PFundComponentDataModel]] = PFundComponentDataModel  # ty: ignore[invalid-mutable-override]
    data_domain: ClassVar[DataCategory] = DataCategory.COMPONENT_DATA
    _required_methods: ClassVar[dict[str, tuple[str, ...]]] = {
        PFundComponentFeedDownloadRequest.extract_type: ("_download_impl",),
    }

    @classmethod
    def create_data_model(
        cls,
        component: Component,
        artifact_type: ArtifactType | str,
        checkpoint_step: int | None = None,
    ) -> PFundComponentDataModel:
        """Create the data model of a component's artifact.

        Args:
            component: the component the artifact belongs to.
            artifact_type: the artifact, e.g. 'data', 'model'.
            checkpoint_step: step of the checkpoint artifact, only for artifact_type='checkpoint'.
        """
        context = component.context
        return cls.DataModel.for_artifact(
            component,
            artifact_type,
            data_source=cls.DataSource.metadata.name,
            env=context.env,
            project_name=context.project_name,
            run_name=context.run_name,
            checkpoint_step=checkpoint_step,
        )

    def _append_request(self, request: PFundComponentFeedBaseRequest) -> None:
        # A component feed has exactly one purpose per run: download/retrieve ONE
        # artifact type. Artifacts can't be combined (a data frame and model/source
        # bytes share no schema), so there's nothing to aggregate across requests.
        # Reject a second request at the source — both download() and retrieve()
        # funnel through here — so run() provably operates on a single dataflow.
        if self._requests:
            current_request = self._get_current_request()
            raise ValueError(
                f"{self.name} runs one artifact per feed: a "
                + f"{current_request.artifact_type} request is already queued. "
                + "Use a fresh feed (or run() the queued one) before requesting "
                + f"{request.artifact_type}."
            )
        return super()._append_request(request)

    def transform(self, *funcs: Callable[..., Any]) -> Self:
        raise NotImplementedError(
            f"{self.name} does not support transform(): artifacts are persisted as-is"
        )

    @requires("download")
    def download(
        self,
        component: Component,
        artifact_type: ArtifactType | str,
        *,
        checkpoint_step: int | None = None,
        io: BaseIO | None = None,
    ) -> Self | RunResult:
        """Extract a component artifact and optionally persist it to storage.

        Args:
            component: the component to extract the artifact from.
            artifact_type: The artifact to extract.
            checkpoint_step: Step identifying the checkpoint artifact. Required
                when ``artifact_type`` is ``checkpoint`` and invalid otherwise.
            io: Storage destination. If omitted, the artifact is only
                returned by the dataflow.

        Returns:
            The feed in pipeline mode; otherwise the completed run result.
        """
        context = component.context
        setup_logging(env=context.env)
        request = cast("type[PFundComponentFeedDownloadRequest]", self.Requests.download)(
            data_source=self.data_source,
            io=io,
            env=context.env,
            project_name=context.project_name,
            run_name=context.run_name,
            component=component,
            artifact_type=artifact_type,
            checkpoint_step=checkpoint_step,
        )
        self._append_request(request)
        self._create_batch_dataflows(
            extract_func=self._download_impl,
            data_models=[request.to_data_model()],
        )
        return self.run() if not self.is_pipeline() else self

    @staticmethod
    def _download_impl(
        request: PFundComponentFeedBaseRequest, data_model: PFundComponentDataModel
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
        checkpoint_step: int | None = None,
        io: BaseIO | None = None,
    ) -> Self | RunResult:
        """Read a component artifact back FROM STORAGE (read-back twin of download+load)."""
        from pfeed.io.parquet_io import ParquetIO

        context = component.context
        setup_logging(env=context.env)
        request = cast("type[PFundComponentFeedRetrieveRequest]", self.Requests.retrieve)(
            data_source=self.data_source,
            io_for_retrieval=io or ParquetIO(),
            env=context.env,
            project_name=context.project_name,
            run_name=context.run_name,
            component=component,
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
    ) -> pl.LazyFrame | None:
        assert isinstance(request, PFundComponentFeedRetrieveRequest)
        lf, metadata = request.create_handler(data_model).read()
        # every partition stores the same component snapshot, see PFundComponentDataHandler.write_batch()
        if metadata:
            data_model.metadata = next(iter(metadata.values()))
        return lf

    def _get_default_transformations(
        self, request: PFundComponentFeedBaseRequest, /
    ) -> list[Callable[..., Any]]:
        # artifacts are persisted and returned as-is
        match request.extract_type:
            case "download" | "retrieve":
                return []
            case _:
                raise ValueError(f"Unknown extract type: {request.extract_type}")

    def run(self, use_prefect: bool = False, **prefect_kwargs: Any) -> RunResult:
        from pfeed.dataflow.result import RunResult

        dataflows = self._run_batch_dataflows(
            use_prefect=use_prefect, prefect_kwargs=prefect_kwargs
        )
        # one component, one artifact type, one request (enforced in _append_request),
        # so there is exactly one dataflow — nothing to aggregate.
        [dataflow] = dataflows
        return RunResult(data=dataflow.result.data, dataflows=dataflows)
