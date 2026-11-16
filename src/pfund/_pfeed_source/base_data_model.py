from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pfund.enums import ComponentType

from pathlib import Path

from pfeed.base.data_model import BaseDataModel
from pfund_blocks.enums.env import Environment
from pydantic import field_validator


class PFundBaseDataModel(BaseDataModel):
    """Identifies a pfund run; paths are relative to the storage's data_path."""

    env: Environment | str
    project_name: str = "default_project"
    run_name: str = "default_run"

    @field_validator("env", mode="before")
    @classmethod
    def _validate_env(cls, v: str | Environment) -> Environment:
        if isinstance(v, str):
            return Environment(v.upper())
        return v

    # also lowered by PFundBaseRequest, here too since create_data_model() doesn't go through a request
    @field_validator("project_name", "run_name", mode="before")
    @classmethod
    def _lower_names(cls, value: str) -> str:
        return value.lower()

    @property
    def project_path(self) -> Path:
        return Path("runs") / f"env={self.env}" / self.project_name.lower()

    @property
    def run_path(self) -> Path:
        return self.project_path / self.run_name.lower()

    @property
    def db_path(self) -> Path:
        return self.run_path / "pfund.db"

    def component_path(self, component_type: ComponentType, component_id: str) -> Path:
        return self.run_path / component_type.to_plural() / component_id
