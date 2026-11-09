from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Self, assert_never, cast

if TYPE_CHECKING:
    from collections.abc import Callable

    from narwhals.typing import IntoFrame

    from pfeed.dataflow.result import DataFlowResult, RunResult
    from pfeed.sources.pfund.requests import (
        PFundComponentFeedDownloadRequest,
        PFundComponentFeedRetrieveRequest,
    )
    from pfund.components.models.model_base import BaseModel
    from pfund.enums import ComponentType
    from pfund.typing import Component

    PFundComponentFeedRequest = (
        PFundComponentFeedDownloadRequest | PFundComponentFeedRetrieveRequest
    )

import polars as pl

from pfeed.config import setup_logging
from pfeed.enums import DataCategory, DataStorage, DataTool, IOFormat
from pfeed.base.feed import BaseFeed
from pfeed.io.io_config import IOConfig
from pfeed.sources.pfund.component_data_model import PFundComponentDataModel
from pfeed.sources.pfund.component_metadata import (
    ComponentMetadata,
    PFundComponentDataMetadata,
    RunMetadata,
)
from pfeed.sources.pfund.source import PFundSource
from pfeed.storages.storage_config import StorageConfig
from pfund.enums import ArtifactType
from pfund_blocks.enums.env import Environment


class PFundComponentFeed(BaseFeed):
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

    DataSource: ClassVar[type[PFundSource]] = PFundSource
    data_source: PFundSource

    DataModel: ClassVar[type[PFundComponentDataModel]] = PFundComponentDataModel
    data_domain: ClassVar[DataCategory] = DataCategory.COMPONENT_DATA

    def __init__(
        self,
        component: Component | None = None,
        pipeline_mode: bool = False,
        num_workers: int | None = None,
    ):
        """
        Args:
            component: the component this feed operates on. Optional rather than
                required because the client builds the feed eagerly, before any
                component exists (`PFund._create_feeds`), then points it via
                `with_component()`. Direct users can pass it up front:
                `PFundComponentFeed(component)`.
            pipeline_mode: see BaseFeed. The feed already defers everything to an
                explicit `download()/load()/run()`, so this just composes with pfeed's
                pipeline machinery.
            num_workers: see BaseFeed. NOTE: only sound for the read path (`retrieve`).
                On `download`, the extract closure is a bound method, so shipping a
                dataflow to a Ray worker pickles the whole feed — including the live
                component. That contradicts the "component is local, only bytes move"
                principle and breaks on unpicklable components, so prefer leaving it
                None for downloads.
        """
        super().__init__(pipeline_mode=pipeline_mode, num_workers=num_workers)
        self._component: Component | None = component

    def with_component(self, component: Component) -> Self:
        """Point the feed at a component (fluent builder setter).

        Conceptually the same as storage's `with_data_model()` — it sets the one
        subject this feed operates on and returns self for chaining, so switching
        components is just `feed.with_component(other).retrieve(...)`.
        """
        self._component = component
        return self

    @property
    def component(self) -> Component:
        assert self._component is not None, "component is not set"
        return self._component

    @property
    def component_type(self) -> ComponentType:
        return self.component.component_type

    def _append_request(self, request: PFundComponentFeedRequest) -> None:
        # A component feed has exactly one purpose per run: download/retrieve ONE
        # artifact type. Artifacts can't be combined (a data frame and model/source
        # bytes share no schema), so there's nothing to aggregate across requests.
        # Reject a second request at the source — both download() and retrieve()
        # funnel through here — so run() provably operates on a single dataflow.
        if self._requests:
            current_request = cast(
                "PFundComponentFeedRequest", self._get_current_request()
            )
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

    def _check_artifact_type(
        self,
        artifact_type: ArtifactType | str,
        checkpoint_step: int | None,
    ) -> ArtifactType:
        artifact_type = ArtifactType[artifact_type.lower()]
        if (
            artifact_type in (ArtifactType.model, ArtifactType.checkpoint)
            and not self.component.is_model()
        ):
            raise ValueError(
                f"{self.component.name} must be a model for model/checkpoint artifacts"
            )
        if artifact_type == ArtifactType.checkpoint:
            if self.component.env != Environment.BACKTEST:
                raise ValueError(
                    "checkpoint artifacts are only available in backtesting"
                )
            if checkpoint_step is None:
                raise ValueError("checkpoint_step is required for a checkpoint")
        else:
            if checkpoint_step is not None:
                raise ValueError("checkpoint_step is only valid for checkpoints")
        return artifact_type

    def _create_artifact_io_config(
        self,
        artifact_type: ArtifactType | str,
    ) -> IOConfig:
        """Create the IO configuration owned by a component artifact type."""
        artifact_type = ArtifactType[artifact_type.lower()]
        io_format = (
            self.component.settings.datalake
            if artifact_type == ArtifactType.data
            else IOFormat.BLOB
        )
        return self._normalize_io_config(IOConfig(io_format=io_format))

    def download(
        self,
        artifact_type: ArtifactType | str,
        storage_config: StorageConfig | None = None,
        checkpoint_step: int | None = None,
    ) -> Self | RunResult:
        """Extract a component artifact and optionally persist it to storage.

        Args:
            artifact_type: The artifact to extract.
            storage_config: Storage destination. If omitted, the artifact is only
                returned by the dataflow.
            checkpoint_step: Step identifying the checkpoint artifact. Required
                when ``artifact_type`` is ``checkpoint`` and invalid otherwise.

        Returns:
            The feed in pipeline mode; otherwise the completed run result.
        """
        from pfeed.sources.pfund.requests import PFundComponentFeedDownloadRequest

        engine_context = self.component.context
        env = engine_context.env
        setup_logging(env=env)
        artifact_type = self._check_artifact_type(artifact_type, checkpoint_step)
        io_config = self._create_artifact_io_config(artifact_type)
        request = PFundComponentFeedDownloadRequest(
            artifact_type=artifact_type,
            env=env,
            project_name=engine_context.project_name,
            run_name=engine_context.run_name,
            storage_config=storage_config,
            io_config=io_config,
            checkpoint_step=checkpoint_step,
        )
        self._append_request(request)
        _ = self._create_batch_dataflows(extract_func=self._download_impl)
        return self.run() if not self.is_pipeline() else self

    def _download_impl(
        self, data_model: PFundComponentDataModel
    ) -> pl.DataFrame | bytes:
        artifact_type = data_model.artifact_type
        if artifact_type == ArtifactType.source:
            return self.component._source_artifact.read_bytes()
        elif artifact_type == ArtifactType.model:
            return cast("BaseModel", self.component)._model_artifact
        elif artifact_type == ArtifactType.data:
            from pfeed.utils.dataframe import convert_dataframe

            return cast(
                pl.DataFrame,
                convert_dataframe(
                    self.component._data_artifact,
                    data_tool=DataTool.polars,
                ),
            )
        elif artifact_type == ArtifactType.checkpoint:
            checkpoint_artifact = cast("BaseModel", self.component)._checkpoint_artifact
            if checkpoint_artifact is None:
                raise RuntimeError("No checkpoint is staged for download")
            return checkpoint_artifact
        else:
            raise ValueError(f"Unsupported artifact type: {artifact_type}")

    def _create_data_model_from_request(
        self, request: PFundComponentFeedRequest
    ) -> PFundComponentDataModel:
        return self.create_data_model(
            artifact_type=request.artifact_type,
            checkpoint_step=request.checkpoint_step,
        )

    def create_data_model(
        self,
        artifact_type: ArtifactType | str,
        checkpoint_step: int | None = None,
    ) -> PFundComponentDataModel:
        engine_context = self.component.context
        artifact_type = ArtifactType[artifact_type.lower()]
        component_snapshot = self.component.to_dict()
        component_type = component_snapshot.pop("component_type")
        run_mode = component_snapshot.pop("run_mode")
        data_start = component_snapshot.pop("data_start")
        data_end = component_snapshot.pop("data_end")
        settings = component_snapshot.pop("settings")
        metadata = PFundComponentDataMetadata(
            component_id=self.component.component_id,
            component_type=str(component_type),
            component=ComponentMetadata(**component_snapshot),
            run=RunMetadata(
                env=str(engine_context.env),
                project_name=engine_context.project_name,
                run_name=engine_context.run_name,
                run_mode=str(run_mode),
                data_start=data_start,
                data_end=data_end,
                settings=settings,
            ),
        )
        artifact_kwargs: dict[str, Any] = {
            "data_source": self.data_source,
            "data_origin": "",
            "env": engine_context.env,
            "project_name": engine_context.project_name,
            "run_name": engine_context.run_name,
            "component_type": component_type,
            "component_id": self.component.component_id,
            "metadata": metadata,
        }
        match artifact_type:
            case ArtifactType.model:
                from pfeed.sources.pfund.component_data_model import ModelArtifact
                from pfund.components.models.jax_model import JAXModel
                from pfund.components.models.pytorch_model import PyTorchModel
                from pfund.components.models.sklearn_model import SKLearnModel

                if isinstance(self.component, SKLearnModel):
                    extension = ".joblib"
                elif isinstance(self.component, (PyTorchModel, JAXModel)):
                    extension = ".safetensors"
                else:
                    raise ValueError(f"Unsupported model type: {type(self.component)}")
                return ModelArtifact(extension=extension, **artifact_kwargs)
            case ArtifactType.data:
                from pfeed.sources.pfund.component_data_model import DataArtifact

                return DataArtifact(
                    replace_where="true"
                    if engine_context.env == Environment.BACKTEST
                    else None,
                    **artifact_kwargs,
                )
            case ArtifactType.source:
                from pfeed.sources.pfund.component_data_model import SourceArtifact

                return SourceArtifact(
                    filename=self.component._source_artifact.name, **artifact_kwargs
                )
            case ArtifactType.checkpoint:
                from pfeed.sources.pfund.component_data_model import CheckpointArtifact
                from pfund.components.models.jax_model import JAXModel
                from pfund.components.models.pytorch_model import PyTorchModel

                if checkpoint_step is None:
                    raise ValueError("checkpoint_step is required for a checkpoint")

                if isinstance(self.component, PyTorchModel):
                    extension = ".pth"
                elif isinstance(self.component, JAXModel):
                    extension = ".pkl"
                else:
                    raise ValueError(
                        f"Unsupported checkpoint model type: {type(self.component)}"
                    )
                return CheckpointArtifact(
                    extension=extension,
                    step=checkpoint_step,
                    **artifact_kwargs,
                )
            case _:
                assert_never(artifact_type)

    def _get_default_transformations_for_download(
        self, request: PFundComponentFeedDownloadRequest
    ) -> list[Callable[..., Any]]:
        from pfeed.utils.dataframe import convert_dataframe
        from pfeed.utils import lambda_with_name

        default_transformations = []
        if request.artifact_type == ArtifactType.data:
            default_transformations.append(
                lambda_with_name(
                    "convert_to_user_df",
                    lambda df: convert_dataframe(df),
                ),
            )
        return default_transformations

    def retrieve(
        self,
        artifact_type: ArtifactType | str,
        storage_config: StorageConfig | None = None,
        checkpoint_step: int | None = None,
    ) -> Self | RunResult:
        """Read a component artifact back FROM STORAGE (read-back twin of download+load)."""
        from pfeed.sources.pfund.requests import PFundComponentFeedRetrieveRequest

        engine_context = self.component.context
        env = engine_context.env
        setup_logging(env=env)
        artifact_type = self._check_artifact_type(artifact_type, checkpoint_step)
        storage_config = self._normalize_storage_config(
            storage_config or StorageConfig()
        )
        io_config = self._create_artifact_io_config(artifact_type)
        request = PFundComponentFeedRetrieveRequest(
            env=env,
            project_name=engine_context.project_name,
            run_name=engine_context.run_name,
            data_source=self.name,
            artifact_type=artifact_type,
            checkpoint_step=checkpoint_step,
            storage_config_for_retrieval=storage_config,
            io_config_for_retrieval=io_config,
        )
        self._append_request(request)
        _ = self._create_batch_dataflows(
            extract_func=lambda data_model: self._retrieve_impl(data_model, request)
        )
        return self.run() if not self.is_pipeline() else self

    def _retrieve_impl(
        self,
        data_model: PFundComponentDataModel,
        request: PFundComponentFeedRetrieveRequest,
    ) -> Any:
        storage_config = request.storage_config_for_retrieval
        io_config = request.io_config_for_retrieval
        Storage = DataStorage[storage_config.storage].storage_class
        storage = Storage.from_storage_config(storage_config).with_io(io_config)
        _ = storage.with_data_model(data_model)
        artifact: pl.LazyFrame | bytes | None = storage.read()
        if artifact is not None:
            metadata_by_path = storage.data_handler.read_metadata()
            if metadata_by_path:
                data_model.metadata = PFundComponentDataMetadata.model_validate(
                    next(iter(metadata_by_path.values())),
                )
            self.logger.debug(f"retrieved artifact {data_model} from {storage}")
        else:
            self.logger.debug(f"no artifact found for {data_model} in {storage}")
        return artifact

    def _get_default_transformations_for_retrieve(
        self, request: PFundComponentFeedRetrieveRequest
    ) -> list[Callable[..., Any]]:
        from pfeed.utils.dataframe import convert_dataframe
        from pfeed.utils import lambda_with_name

        default_transformations = []
        if request.artifact_type == ArtifactType.data:
            default_transformations.append(
                lambda_with_name(
                    "convert_to_user_df",
                    lambda df: convert_dataframe(df),
                ),
            )
        return default_transformations

    # TODO: connect to mtflow's ws server
    def stream(self, *args: Any, **kwargs: Any) -> Self:
        # streaming a component's live signals — separate (StreamingFeedMixin) path
        raise NotImplementedError(f"{self.name} stream() is not implemented yet")

    def run(self, use_prefect: bool = False, **prefect_kwargs: Any) -> RunResult:
        from pfeed.utils.dataframe import convert_dataframe
        from pfeed.dataflow.result import RunResult

        dataflows = self._run_batch_dataflows(use_prefect=use_prefect, prefect_kwargs=prefect_kwargs)
        # one component, one artifact type, one request (enforced in _append_request),
        # so there is exactly one dataflow — nothing to aggregate.
        [dataflow] = dataflows
        result: DataFlowResult = dataflow.result
        data: IntoFrame | bytes | None = result.data
        # NOTE: only data artifact returns dataframe
        is_dataframe = data is not None and not isinstance(data, bytes)
        if is_dataframe:
            data = convert_dataframe(data)
        return RunResult(data=data, dataflows=dataflows)
