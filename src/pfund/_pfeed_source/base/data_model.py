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

    @property
    def project_path(self) -> Path:
        return Path("runs") / f"env={self.env}" / self.project

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
