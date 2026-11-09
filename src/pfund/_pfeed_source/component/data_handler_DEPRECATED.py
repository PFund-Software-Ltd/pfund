from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, assert_never, cast

if TYPE_CHECKING:
    from narwhals.typing import IntoFrame

    from pfeed.io.database_io import DBPath
    from pfeed.io.file_io import FileIO
    from pfeed.sinks.base_sink import BaseSink
    from pfeed.sources.pfund.component_data_model import PFundComponentDataModel

import polars as pl

from pfeed.base.data_handler import BaseDataHandler
from pfeed.enums import DataLayer, DataTool, IOType
from pfeed.io.table_io import TablePath
from pfeed.sources.pfund.component_metadata import PFundComponentDataMetadata
from pfeed.utils.file_path import FilePath
from pfund.enums import ArtifactType, ComponentType
from pfund_blocks.enums.env import Environment


class PFundComponentDataHandler(BaseDataHandler):
    """Persists pfund component artifacts on file-based storage.

    The artifact arrives already in its owner's currency — opaque bytes for
    model/source (serialized by pfund), a frame for `data` — so this handler does
    no (de)serialization itself: it just routes the payload to the IO at a path
    derived from the component's identity. With the blob path (IOFormat.BLOB -> FileIO)
    the payload is bytes; the IO writes/reads them verbatim.

    Storage Layout:
    data_path/  (e.g. pfund's config.data_path)
        runs/
            env=BACKTEST or SANDBOX or PAPER or LIVE/
                {project_name}/
                    trackio.db  (when mtflow is used)
                    optuna.db  (optional, when mtflow is used)
                    default_run/  (or run_00x when mtflow is used)
                        pfund.db (engine states (e.g. used to match back client order ids),
                                  also used to host a fake server in SANDBOX trading)
                        {component_type}/
                            {component_class_name-component_id}/
                                artifacts, e.g.
                                - trading_df.delta  (delta table of trading df (df in pfud's trading store))
                                - metadata.json
                                - model.safetensors, model.joblib (trained model)
                                - {component}.py  (component source code)
                                checkpoints/
                                    step=0/
                                        checkpoint.pth or checkpoint.pkl
    """

    _data_model: PFundComponentDataModel
    Metadata: ClassVar[type[PFundComponentDataMetadata]] = PFundComponentDataMetadata
    PARTITION_COLUMNS: ClassVar[list[str]] = ["chunk_num"]
    IO_USING_PARTITION_COLUMNS: ClassVar[set[str]] = {"DeltaLakeIO"}

    def __init__(
        self,
        data_path: FilePath,
        data_layer: DataLayer,  # not in use
        data_domain: str,  # not in use
        data_model: PFundComponentDataModel,
        io: FileIO,
        sink: BaseSink | None = None,
    ):
        super().__init__(
            data_path=data_path,
            data_layer=data_layer,
            data_domain=data_domain,
            data_model=data_model,
            io=io,
            sink=sink,
        )
        if self._io_type == IOType.FILE:
            self._file_paths = [self._create_file_path()]
        elif self._io_type == IOType.TABLE and self._table_path:
            artifact = self._data_model
            if artifact.artifact_type == ArtifactType.data:
                self._table_path /= f"trading_df{artifact.extension}"

    @property
    def file_path(self) -> FilePath:
        return self._file_paths[0]

    def _create_file_path(self) -> FilePath:
        artifact = self._data_model
        artifact_dir = self._create_table_path()
        match artifact.artifact_type:
            case ArtifactType.source:
                from pfeed.sources.pfund.component_data_model import SourceArtifact

                return artifact_dir / cast(SourceArtifact, artifact).filename
            case ArtifactType.model:
                return artifact_dir / f"model{artifact.extension}"
            case ArtifactType.checkpoint:
                from pfeed.sources.pfund.component_data_model import CheckpointArtifact

                return (
                    artifact_dir
                    / "checkpoints"
                    / f"step={cast(CheckpointArtifact, artifact).step}"
                    / f"checkpoint{artifact.extension}"
                )
            case ArtifactType.data:
                raise ValueError("data artifacts use a table path, not a file path")
            case _:
                assert_never(artifact.artifact_type)

    def _create_table_path(self) -> TablePath:
        from pfund.engines.base_engine import BaseEngine

        artifact = self._data_model
        run_path = BaseEngine._create_run_path(
            data_path=cast(FilePath, self._data_path),
            env=Environment(artifact.env),
            project_name=artifact.project_name,
            run_name=artifact.run_name,
        )
        return TablePath(
            run_path
            / ComponentType[artifact.component_type].to_plural()
            / artifact.component_id
        )

    def _create_db_path(self, *args: Any, **kwargs: Any) -> DBPath:
        raise NotImplementedError

    def _requires_partitioning(self) -> bool:
        """Partition standalone strategy backtests within one data artifact.

        ``chunk_num`` is a pfund-owned dataframe column. Pfeed only gives that
        logical result boundary a physical representation when persisting a
        strategy's BACKTEST data to a table format that supports partitioning.
        Model/feature data and strategy data from other environments retain
        their existing unpartitioned layout.
        """
        artifact = self._data_model
        return (
            artifact.artifact_type == ArtifactType.data
            and artifact.env == Environment.BACKTEST
            and artifact.component_type == ComponentType.strategy
            and super()._requires_partitioning()
        )

    def _validate_partition_columns(self, df: pl.DataFrame) -> None:
        missing_columns = [
            column for column in self.PARTITION_COLUMNS if column not in df.columns
        ]
        if missing_columns:
            raise ValueError(
                "BACKTEST strategy data is missing required partition "
                + f"column(s): {missing_columns}"
            )

        chunk_num = df.get_column("chunk_num")
        if not chunk_num.dtype.is_integer():
            raise TypeError(
                "BACKTEST strategy data column 'chunk_num' must have an integer "
                + f"dtype, got {chunk_num.dtype}"
            )
        if chunk_num.null_count() > 0:
            raise ValueError(
                "BACKTEST strategy data column 'chunk_num' cannot contain nulls"
            )
        if not chunk_num.is_empty() and cast(int, chunk_num.min()) < 0:
            raise ValueError(
                "BACKTEST strategy data column 'chunk_num' cannot contain negative values"
            )

    def write_batch(self, data: IntoFrame | bytes) -> None:
        with self.io:
            if isinstance(data, bytes):
                # source (.py) and model weights (.safetensors / .joblib) arrive as
                # opaque bytes pfund already serialized; FileIO writes them verbatim.
                self.io.write(data, self.file_path)
                if self._data_model.artifact_type == ArtifactType.source:
                    self.io.write_metadata(
                        self.file_path, metadata=self._create_metadata()
                    )
            else:  # data artifact writing to deltalake
                from pfeed.utils.dataframe import convert_dataframe
                from pfeed.sources.pfund.component_data_model import DataArtifact

                table_path = self._table_path
                assert table_path is not None, "table path is not initialized"
                artifact = cast(DataArtifact, self._data_model)
                lf = cast(pl.LazyFrame, convert_dataframe(data, DataTool.polars))
                df = lf.collect()
                write_kwargs: dict[str, Any] = {}
                if self._requires_partitioning():
                    self._validate_partition_columns(df)
                    write_kwargs["partition_by"] = self.PARTITION_COLUMNS
                self.io.write(
                    df.to_arrow(),
                    table_path,
                    delete_where=artifact.replace_where,
                    **write_kwargs,
                )
                self.io.write_metadata(table_path, metadata=self._create_metadata())

    def read(self) -> pl.LazyFrame | bytes | None:
        match self._io_type:
            case IOType.FILE:
                return self.io.read(file_paths=self._file_paths)
            case IOType.TABLE:
                table_path = self._table_path
                assert table_path is not None, "table path is not initialized"
                return self.io.read(table_path)
            case IOType.DATABASE:
                raise NotImplementedError(
                    f"{self.__class__.__name__} does not support database IO"
                )
            case _:
                assert_never(self._io_type)

    # REVIEW: no validation, not in use
    def _validate_schema(self, df: pl.LazyFrame) -> Any:
        return df

    def _create_metadata(self) -> PFundComponentDataMetadata:
        metadata = self._data_model.metadata
        if metadata is None:
            raise ValueError("component metadata is required when writing an artifact")
        return metadata
