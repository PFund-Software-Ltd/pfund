from __future__ import annotations

from pfeed.base.request import BaseRequest
from pfund_blocks.enums.env import Environment
from pydantic import field_validator


class PFundBaseRequest(BaseRequest):
    """Base request of pfund's feeds, identifies the engine run the data belongs to."""

    env: Environment
    project_name: str
    run_name: str

    @field_validator("env", mode="before")
    @classmethod
    def _validate_env(cls, v: Environment | str) -> Environment:
        if isinstance(v, str):
            return Environment(v.upper())
        return v

    @field_validator("project_name", mode="before")
    @classmethod
    def _lower_project_name(cls, value: str) -> str:
        return value.lower()

    @field_validator("run_name", mode="before")
    @classmethod
    def _lower_run_name(cls, value: str) -> str:
        return value.lower()
