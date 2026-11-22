from __future__ import annotations

from typing import ClassVar

from pathlib import Path

from pfeed.base.data_model import BaseDataModel
from pfund_blocks.enums.env import Environment
from pydantic import field_validator


class PFundBaseDataModel(BaseDataModel):
    """Identifies a pfund run; paths are relative to the storage's data_path."""

    DEFAULT_PROJECT_NAME: ClassVar[str] = "default_project"
    DEFAULT_RUN_NAME: ClassVar[str] = "default_run"
    # the blob IO's dir under the lakehouse path, see TradingStore.setup_lakehouse()
    RUNS_DIR_NAME: ClassVar[str] = "runs"

    env: Environment | str
    project: str = DEFAULT_PROJECT_NAME
    run: str = DEFAULT_RUN_NAME

    @field_validator("env", mode="before")
    @classmethod
    def _validate_env(cls, v: str | Environment) -> Environment:
        return Environment(v.upper())

    # also lowered by PFundBaseRequest, here too since create_data_model() doesn't go through a request
    @field_validator("project", "run", mode="before")
    @classmethod
    def _lower_names(cls, value: str) -> str:
        return value.lower()

    # NOTE: plain dirs, the same as the namespace of every pfund DatasetKey written by the blob IO (hive_dataset_dirs=False),
    # see PFundComponentDataHandler._create_dataset_key(), so a run's files share one run dir
    @property
    def project_path(self) -> Path:
        return Path(self.RUNS_DIR_NAME) / str(self.env) / self.project

    @property
    def run_path(self) -> Path:
        return self.project_path / self.run

    def __str__(self) -> str:
        return ":".join(
            [
                self.env,
                super().__str__(),
                self.project,
                self.run,
            ]
        )
