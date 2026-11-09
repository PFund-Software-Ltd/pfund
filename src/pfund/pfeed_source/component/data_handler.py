from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

if TYPE_CHECKING:
    from pfeed.io.base_io import BaseIO, Partition
    from pfeed.streaming.sink import Sink

    from pfeed_pfund.component.data_model import PFundComponentDataModel

import datetime

import polars as pl
from pfeed.base.data_handler import BaseDataHandler, BaseDataMetadata
from pfeed.enums import DataCategory, DataLayer
from pfeed.io.base_io import DatasetKey
from pfund.enums import ArtifactType, ComponentType
from pfund_blocks.enums.env import Environment
from pydantic import BaseModel, Field


# FIXME: restructure it in pfeed-pfund
def _utc_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)


class ComponentMetadata(BaseModel):
    """A persisted snapshot of a pfund component's definition."""

    class_name: str
    source_sha256: str
    component_name: str
    resolution: str
    df_form: Literal["wide", "long"]
    signal_cols: list[str] = Field(default_factory=list)
    model: str | None = None
    signature: tuple[tuple[Any, ...], dict[str, Any]]
    config: dict[str, Any]
    params: dict[str, Any]
    datas: list[dict[str, Any]]
    strategies: list[str] = Field(default_factory=list)
    models: list[str] = Field(default_factory=list)
    features: list[str] = Field(default_factory=list)


class RunMetadata(BaseModel):
    """The run context in which the component snapshot was produced."""

    env: str
    project_name: str
    run_name: str
    run_mode: str
    data_start: datetime.date
    data_end: datetime.date
    settings: dict[str, Any]


class PFundComponentDataMetadata(BaseDataMetadata):
    """Per-partition metadata of a component artifact: the component snapshot that produced it.

    The run identity (env, project, run) and the component identity (type, id) are in the DatasetKey,
    they are repeated in the snapshot since it is the component's full definition, read back as one unit.
    """

    schema_version: Literal[1] = 1
    created_at: datetime.datetime = Field(default_factory=_utc_now, frozen=True)
    component_id: str
    component_type: str
    component: ComponentMetadata
    run: RunMetadata


class PFundComponentDataHandler(BaseDataHandler["PFundComponentDataModel", PFundComponentDataMetadata]):
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

    Metadata = PFundComponentDataMetadata
    # pfund-owned column splitting a standalone strategy backtest's data into chunks
    CHUNK_COL: ClassVar[str] = "chunk_num"

    def __init__(
        self,
        data_model: PFundComponentDataModel,
        io: BaseIO,
        # component data has no data layers, data_layer is not part of its DatasetKey
        data_layer: DataLayer | str = DataLayer.CLEANED,
        data_domain: str = DataCategory.COMPONENT_DATA,
        sink: Sink | None = None,
    ):
        super().__init__(data_model, io, data_domain=data_domain, data_layer=data_layer, sink=sink)
        # FIXME: BaseIO only stores frames, source/model/checkpoint artifacts are opaque bytes and need a bytes IO
        if data_model.artifact_type != ArtifactType.data:
            raise NotImplementedError(f'storing {data_model.artifact_type} artifacts is not supported yet, no IO stores bytes')
        if sink is not None:
            raise NotImplementedError('storing streamed component data is not supported yet')

    def _requires_partitioning(self) -> bool:
        """Partition standalone strategy backtests within one data artifact.

        ``chunk_num`` is a pfund-owned dataframe column. Pfeed only gives that
        logical result boundary a physical representation when persisting a
        strategy's BACKTEST data to a table format that supports partitioning.
        Model/feature data and strategy data from other environments retain
        their existing unpartitioned layout.
        """
        data_model = self._data_model
        return data_model.env == Environment.BACKTEST and data_model.component_type == ComponentType.strategy

    def _create_dataset_key(self) -> DatasetKey:
        data_model = self._data_model
        return DatasetKey(
            namespace={
                'env': str(data_model.env),
                'project_name': data_model.project_name,
                'run_name': data_model.run_name,
            },
            name={
                'component_type': str(data_model.component_type),
                'component_id': data_model.component_id,
                'artifact_type': str(data_model.artifact_type),
            },
            partition_by=(self.CHUNK_COL,) if self._requires_partitioning() else (),
        )

    def _validate_schema(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self._requires_partitioning():
            return df
        if self.CHUNK_COL not in df.columns:
            raise ValueError(f"BACKTEST strategy data is missing the required partition column '{self.CHUNK_COL}'")
        chunk_num = df.get_column(self.CHUNK_COL)
        if not chunk_num.dtype.is_integer():
            raise TypeError(f"BACKTEST strategy data column '{self.CHUNK_COL}' must have an integer dtype, got {chunk_num.dtype}")
        if chunk_num.null_count() > 0:
            raise ValueError(f"BACKTEST strategy data column '{self.CHUNK_COL}' cannot contain nulls")
        if not chunk_num.is_empty() and cast('int', chunk_num.min()) < 0:
            raise ValueError(f"BACKTEST strategy data column '{self.CHUNK_COL}' cannot contain negative values")
        return df

    def _create_metadata(self) -> PFundComponentDataMetadata:
        metadata = self._data_model.metadata
        if metadata is None:
            raise ValueError('component metadata is required when writing an artifact')
        return metadata

    def write_batch(self, df: pl.DataFrame) -> None:
        """Writes the component's data.

        A backtest reruns from scratch, so its data replaces the whole dataset,
        including the chunks of a previous run that this run no longer has.
        Data of other envs (e.g. LIVE) accumulates across sessions, so it is appended.
        """
        df = self._validate_schema(df)
        metadata = self._create_metadata()
        if self._requires_partitioning():
            partitions: set[Partition] = {(chunk_num,) for chunk_num in df.get_column(self.CHUNK_COL).unique()}
        else:
            partitions = {()}
        if self._data_model.env == Environment.BACKTEST:
            # existing partitions not in df are written as empty partitions, i.e. their rows are deleted
            _, existing = self._read()
            self._write(df, {partition: metadata for partition in partitions | existing.keys()}, mode='replace')
        else:
            self._write(df, {partition: metadata for partition in partitions}, mode='append')

    def read(self) -> tuple[pl.LazyFrame | None, dict[Partition, PFundComponentDataMetadata]]:
        return self._read()
