from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, assert_never, cast

if TYPE_CHECKING:
    from pfeed.io.base_io import Partition

    from pfund._pfeed_source.component.data_model import (
        CheckpointArtifact,
        ModelArtifact,
        PFundComponentDataModel,
        SourceArtifact,
    )

import datetime
import json
from functools import cached_property

import polars as pl
from pfeed.base.data_handler import BaseDataMetadata
from pfeed.enums import DataLayer
from pfeed.io.base_io import BaseIO, DatasetKey
from pfeed.io.blob_io import BlobIO
from pfund_kit.utils.temporal import get_utc_now
from pydantic import Field

from pfund._pfeed_source.base.data_handler import PFundBaseDataHandler
from pfund._pfeed_source.data_category import PFundDataCategory
from pfund.enums import ArtifactType


class PFundComponentDataMetadata(BaseDataMetadata):
    """pfeed's per-partition metadata of a component artifact, i.e. the IO's commit marker.

    The component's own metadata is written once in pfund's metadata.json, not per artifact, see _write_component_metadata().
    """

    created_at: datetime.datetime = Field(default_factory=get_utc_now, frozen=True)
    # hash of the code that wrote the artifact (component._source_sha256), stamped per artifact, not in metadata.json:
    # metadata.json is rewritten every run, the artifact only when it is rewritten, so a match means it came from this code.
    # None for an artifact written before it was stamped, i.e. unknown code
    source_sha256: str | None = None
    # the engine's data range the artifact was written in (component.context.data_start/data_end), stamped for the
    # same reason: a trading df computed on another range is stale, see TradingStore.materialize()
    data_start: datetime.date | None = None
    data_end: datetime.date | None = None
    # pfund's metadata.json of the component, attached by PFundComponentDataHandler.read(),
    # excluded from model_dump(), so it is never written into the per-partition metadata
    component: dict[str, Any] | None = Field(default=None, exclude=True)


class PFundComponentDataHandler(
    PFundBaseDataHandler["PFundComponentDataModel", PFundComponentDataMetadata]
):
    """Persists pfund component artifacts on file-based storage.

    The artifact arrives already in its owner's currency — opaque bytes for
    source/model/checkpoint (serialized by pfund), a frame for `data` — so this handler does
    no (de)serialization itself: it routes the payload to the IO under the component's DatasetKey.
    A frame goes to a BaseIO (the component's own datalake), bytes to a BlobIO, which writes/reads them verbatim;
    a checkpoint is the partition of its step.

    Storage Layout:
    <lakehouse_path>/  (pfund's engine-scoped config.data_path by default, see TradingStore.setup_lakehouse())
        runs/  (the store's BlobIO, plain dataset dirs, i.e. hive_dataset_dirs=False)
            BACKTEST or SANDBOX or PAPER or LIVE/
                {project}/
                    trackio.db  (when mtflow is used)
                    optuna.db  (optional, when mtflow is used)
                    default_run/  (or run_00x when mtflow is used)
                        pfund.db (engine states (e.g. used to match back client order ids),
                                  also used to host a fake server in SANDBOX trading)
                        {component_type plural}/  (e.g. strategies/)
                            {component_id}/  ({component_class_name}-{hash}, see PFundComponentDataModel.component_path)
                                _pfeed_metadata.json  (BlobIO's manifest of the files below)
                                metadata.json  (pfund's own metadata of the run)
                                {component}.py  (component source code)
                                model.safetensors or model.joblib  (trained model)
                                trading_df/  (the component's own DuckLakeIO, catalog pfeed.db + data/ inside,
                                              see TradingStore.setup_lakehouse())
                                checkpoints/
                                    step=0/
                                        _pfeed_metadata.json
                                        checkpoint.pth
    """

    Metadata = PFundComponentDataMetadata
    # NOTE: don't confuse them: Metadata above is pfeed's per-partition metadata (the IO's commit marker, _pfeed_metadata.json),
    # the ones below are for pfund's metadata.json of the component, written once per run per component
    METADATA_FILE_NAME: ClassVar[str] = "metadata.json"
    METADATA_SCHEMA_VERSION: ClassVar[int] = 1
    CHECKPOINTS_DIR_NAME: ClassVar[str] = "checkpoints"

    def __init__(
        self,
        data_model: PFundComponentDataModel,
        io: BaseIO | BlobIO,
        # NOTE: component data has no data layers, data_layer is not part of its DatasetKey
        data_layer: DataLayer | str = DataLayer.CLEANED,
        data_domain: str = PFundDataCategory.COMPONENT_DATA,
        # NOTE: always passed by pfeed's dataflow; streaming isn't supported, setup_streaming() raises
        is_streaming: bool = False,
    ):
        super().__init__(
            data_model,
            io,
            data_domain=data_domain,
            data_layer=data_layer,
            is_streaming=is_streaming,
        )
        self._validate_io()

    @cached_property
    def _component_metadata_io(self) -> BlobIO:
        """pfund's metadata.json is a blob, always stored in the store's BlobIO, independent of the artifact's io.

        It is the BlobIO of the component's other bytes artifacts, metadata.json sits next to them
        in the component's dir, listed in the same manifest (see TradingStore.setup_lakehouse()).
        """
        return self._data_model.component.store.blob_io

    def _validate_io(self) -> None:
        """Raises TypeError if the IO can't store the artifact: bytes need a BlobIO, frames a BaseIO."""
        artifact_type = self._data_model.artifact_type
        if self._should_use_blob_io() and not isinstance(self._io, BlobIO):
            raise TypeError(
                f"{artifact_type} artifacts are bytes, they need a BlobIO, got {self._io}"
            )
        if not self._should_use_blob_io() and not isinstance(self._io, BaseIO):
            raise TypeError(
                f"{artifact_type} artifacts are frames, they need a BaseIO, got {self._io}"
            )

    def _should_use_blob_io(self) -> bool:
        """Whether the artifact is bytes (source/model/checkpoint), stored in a BlobIO."""
        return self._data_model.artifact_type != ArtifactType.data

    @property
    def _blob_io(self) -> BlobIO:
        if not isinstance(self._io, BlobIO):
            raise TypeError(f"{self._io} doesn't store bytes, use a BlobIO")
        return self._io

    def _blob_partition(self) -> Partition:
        """The bytes artifact's partition: the data model's values of the key's partition levels, e.g. (step,) for a checkpoint.

        A blob's partition level is named after a field of its data model, e.g. "step" -> CheckpointArtifact.step,
        so the levels are only decided in _create_dataset_key().
        """
        if not self._should_use_blob_io():
            raise TypeError(
                "data artifacts are frames, their partitions come from the frame's columns"
            )
        return tuple(
            getattr(self._data_model, level)
            for level in self._dataset_key.partition_columns
        )

    def _blob_file_name(self) -> str:
        """The bytes artifact's file name in storage, e.g. 'model.joblib', so it can be opened as is."""
        data_model = self._data_model
        match data_model.artifact_type:
            case ArtifactType.source:
                return cast("SourceArtifact", data_model).filename
            case ArtifactType.model:
                return f"model{cast('ModelArtifact', data_model).extension}"
            case ArtifactType.checkpoint:
                return f"checkpoint{cast('CheckpointArtifact', data_model).extension}"
            case ArtifactType.data:
                raise TypeError("data artifacts are frames, they have no file name")
            case _:
                assert_never(data_model.artifact_type)

    def _create_dataset_key(self) -> DatasetKey:
        data_model = self._data_model
        component = data_model.component
        # NOTE: chunk_num is not a partition: a backtest replaces the whole dataset anyway (see write_batch()),
        # and the chunks are written together, once, see BacktestEngine._backtest()
        name = {
            "component_type": component.component_type.to_plural(),
            "component_id": component.component_id,
        }
        # the other bytes artifacts are files of the component's dir (the same dataset, see _blob_file_name()),
        # checkpoints are their own dataset in the component's checkpoints/ dir, partitioned by step
        partition_by: tuple[str, ...] = ()
        if data_model.artifact_type == ArtifactType.checkpoint:
            name["artifact"] = self.CHECKPOINTS_DIR_NAME
            partition_by = ("step",)
        return DatasetKey(
            namespace={
                "env": str(data_model.env),
                "project": data_model.project,
                "run": data_model.run,
            },
            name=name,
            partition_by=partition_by,
        )

    def _create_component_dataset_key(self) -> DatasetKey:
        """The key of pfund's metadata.json: the component's dir, where the source and model files are too."""
        key = self._dataset_key
        name = {k: v for k, v in key.name.items() if k != "artifact"}
        return DatasetKey(namespace=key.namespace, name=name, partition_by=())

    def _validate_schema(self, df: pl.DataFrame) -> pl.DataFrame:
        """Checks only what storage relies on: the date column read() sorts by.

        The frame's other columns are owned by the component that produced it (e.g. the backtest kernels).
        chunk_num and product are sort keys too, but optional: chunk_num only exists in a strategy's BACKTEST data,
        product only in long form, see _sort().
        """
        from pfund.components.bar_dataframe import INDEX_COL

        if INDEX_COL not in df.columns:
            raise ValueError(
                f"{self._data_model.artifact_type} artifact is missing the required column '{INDEX_COL}'"
            )
        return df

    @staticmethod
    def _sort_data_artifact(lf: pl.LazyFrame) -> pl.LazyFrame:
        """Sorts the data by the sort keys it has, in order: chunk_num, date, product.

        No IO guarantees the order of the rows it reads, e.g. LIVE data accumulated from many appends.
        chunk_num first, so each chunk (a standalone backtest) reads as its own timeline;
        product last, so the rows of a date in long form come in a deterministic order.
        Only the schema is read, so lf stays lazy.
        """
        from pfund.backtest._mixins.strategy_mixin import BacktestStrategyMixin
        from pfund.components.bar_dataframe import INDEX_COL, PIVOT_COLS

        schema = lf.collect_schema()
        sort_cols = [
            col
            for col in (BacktestStrategyMixin.CHUNK_COL, INDEX_COL, PIVOT_COLS[0])
            if col in schema
        ]
        return lf.sort(sort_cols, maintain_order=True) if sort_cols else lf

    def _create_metadata(self) -> PFundComponentDataMetadata:
        component = self._data_model.component
        return PFundComponentDataMetadata(
            source_sha256=component._source_sha256,
            data_start=component.context.data_start,
            data_end=component.context.data_end,
        )

    def _write_component_metadata(self) -> None:
        """Writes pfund's metadata.json of the component (what it was in its run), replacing the stored one.

        Written once per run per component, as component.to_dict() + a few fields about the record and the run.
        A blob in the component's dir, listed in the dir's _pfeed_metadata.json (BlobIO's manifest) with the other files.
        """
        component_metadata = {
            # bump when a reader would misread an older metadata.json, e.g. a field renamed or removed
            "schema_version": self.METADATA_SCHEMA_VERSION,
            # isoformat(): default=str below would write it with a space instead of the ISO "T"
            "created_at": get_utc_now().isoformat(),
            # in backtesting, it also has backtest_mode, see BacktestMixin.to_dict()
            **self._data_model.component.to_dict(),
        }
        # default=str: the signature/config/params can hold objects JSON can't express, e.g. a torch module,
        # stored as their str, same as component_id hashes them
        blob = json.dumps(component_metadata, default=str).encode()
        self._component_metadata_io.write(
            self._create_component_dataset_key(),
            {(): (blob, self._create_metadata().model_dump(mode="json"))},
            file_name=self.METADATA_FILE_NAME,
        )

    def _read_component_metadata(self) -> dict[str, Any] | None:
        """Reads pfund's metadata.json of the component as written, None if it isn't stored.

        Values come back as JSON types, e.g. dates as ISO strings.
        """
        key = self._create_component_dataset_key()
        blobs, _ = self._component_metadata_io.read(
            key, partitions=[()], file_name=self.METADATA_FILE_NAME
        )
        if () not in blobs:
            return None
        return json.loads(blobs[()])

    # NOTE: named df to match BaseDataHandler.write_batch(), but a bytes artifact is bytes
    def write_batch(self, data: pl.DataFrame | bytes) -> None:
        """Writes the component's artifact, and pfund's metadata.json with the source artifact.

        Always replaces the stored artifact (of its step, for a checkpoint), same as pfeed's write_batch():
        a data artifact replaces the whole dataset, e.g. a rerun with fewer chunks leaves none of the previous run's chunks behind.
        Appending new rows, e.g. live trading data, is write_stream()'s job.
        """
        if self._should_use_blob_io():
            if not isinstance(data, bytes):
                raise TypeError(
                    f"{self._data_model.artifact_type} artifact must be bytes, got {type(data)}"
                )
            metadata = self._create_metadata().model_dump(mode="json")
            self._blob_io.write(
                self._dataset_key,
                {self._blob_partition(): (data, metadata)},
                file_name=self._blob_file_name(),
            )
            # the source is written once per run per component, at the end of the component's build (see _build()),
            # when its snapshot is final, so pfund's metadata.json is written with it
            if self._data_model.artifact_type == ArtifactType.source:
                self._write_component_metadata()
            return
        if not isinstance(data, pl.DataFrame):
            raise TypeError(
                f"data artifact must be a polars DataFrame, got {type(data)}"
            )
        df = self._validate_schema(data)
        self._write(df, {(): self._create_metadata()}, mode="replace")

    # TODO
    def write_stream(self, msg: Any) -> None:
        """Buffers new rows of the data artifact, appended to it by pfeed's flush(), see BaseDataHandler.setup_streaming().

        Not implemented yet, raises NotImplementedError.
        """
        # TODO: use this to write live trading data to the trading df (data artifact), once the live trading flow is decided,
        # e.g. streaming data -> TradingStore updates its df in memory -> the new rows -> write_stream() -> buffered, appended by flush().
        # needs _build_streaming_schema() (the trading df's columns depend on the component, e.g. from store._df after _build())
        # and _stream_partitions() (just {(): metadata}, the data artifact is unpartitioned),
        # and an IO that supports append (e.g. DuckLakeIO, not ParquetIO)
        raise NotImplementedError(
            "writing live trading data to the data artifact is not supported yet"
        )

    def read(
        self,
    ) -> tuple[
        pl.LazyFrame | bytes | None, dict[Partition, PFundComponentDataMetadata]
    ]:
        """Reads the component's artifact: a frame for data, bytes for the others (of its step, for a checkpoint).

        Each partition's metadata comes with pfund's metadata.json of the component attached (see its `component`),
        None if metadata.json isn't stored, e.g. a child's data artifact written before its source (see _build()).

        Returns (None, {}) if it isn't stored.
        """
        data: pl.LazyFrame | bytes | None
        if not self._should_use_blob_io():
            data, metadata = self._read()
            if data is not None:
                data = self._sort_data_artifact(data)
        else:
            partition = self._blob_partition()
            blobs, blob_metadata = self._blob_io.read(
                self._dataset_key,
                partitions=[partition],
                file_name=self._blob_file_name(),
            )
            if partition not in blobs:
                return None, {}
            data = blobs[partition]
            metadata = {partition: self._parse_metadata(blob_metadata[partition])}
        if metadata:
            component_metadata = self._read_component_metadata()
            for md in metadata.values():
                md.component = component_metadata
        return data, metadata
