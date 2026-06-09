# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypedDict

if TYPE_CHECKING:
    import datetime

    from pfund.engines.settings.base_engine_settings import BaseEngineSettings
    from pfund.config import PFundConfig
    from pfund.engines.component_registry import RegistryProxy

from pfeed.enums import DataTool

from pfund.config import get_config, get_logging_config
from pfund.datas.resolution import Resolution
from pfund.engines.component_registry import ComponentRegistry
from pfund.enums import Environment, RunMode, RunStage


class DataRangeDict(TypedDict, total=False):
    start_date: str
    end_date: str


class EngineContext:
    def __init__(
        self,
        env: Environment,
        name: str,
        data_range: str | Resolution | DataRangeDict | Literal["ytd"],
        settings: BaseEngineSettings | None = None,
    ):
        import pfeed as pe

        self.env = env
        self.name = name
        self._env_var_prefix = f"PFUND_{env.upper()}_"
        self._env_vars = self._load_env_vars()
        self.run_mode = self._detect_run_mode()
        self.run_stage = (
            RunStage.experiment
            if self.env == Environment.BACKTEST
            else RunStage.deployment
        )
        self.project_name = "default"
        self.data_start, self.data_end = self._parse_data_range(data_range)
        # NOTE: config obtained by get_config() inside ray actor could be different from the one in the main thread (e.g. after calling pf.configure())
        # so we create the config object here in the context and treat it as the source of truth
        self.pfund_config: PFundConfig = get_config(engine_name=self.name)
        self.pfeed_config = pe.get_config()
        if self.pfeed_config.data_tool not in [DataTool.pandas, DataTool.polars]:
            raise ValueError(f"Unsupported data tool: {self.pfeed_config.data_tool}")
        self.logging_config: dict[str, Any] = get_logging_config()
        self.settings = self._resolve_settings(settings)
        # starts local; upgraded in-place to the shared actor-backed registry the first
        # time a remote component is declared (see component_registry.to_registry_proxy)
        self.component_registry: ComponentRegistry | RegistryProxy = ComponentRegistry()

    # REVIEW: engine has no REMOTE run mode
    @staticmethod
    def _detect_run_mode() -> RunMode:
        import sys

        if sys.platform == "emscripten":
            return RunMode.WASM
        else:
            return RunMode.LOCAL

    def _resolve_settings(
        self, settings: BaseEngineSettings | None
    ) -> BaseEngineSettings:
        if settings is None:
            return self._load_settings()
        if settings.persist:
            self._save_settings(settings)
        return settings

    def set_run_stage(self, stage: RunStage | str) -> None:
        stage = RunStage[stage.upper()]
        if self.env == Environment.BACKTEST:
            assert stage in [RunStage.experiment, RunStage.refinement], (
                "Run stage can only be set to EXPERIMENT or REFINEMENT in backtesting"
            )
        else:
            assert stage == RunStage.deployment, (
                "Run stage can only be set to DEPLOYMENT in trading"
            )
        self.run_stage = stage

    def set_project_name(self, project_name: str) -> None:
        self.project_name = project_name.lower()

    def _load_env_vars(self) -> dict[str, str]:
        from dotenv import dotenv_values, find_dotenv

        # load env vars manually to avoid loading into os.environ
        # NOTE: this allows multiple engines with different envs in the same process
        env_filename = f".env.{self.env.lower()}"
        env_file_path = find_dotenv(
            filename=env_filename, usecwd=True, raise_error_if_not_found=False
        )
        raw_env_vars = dotenv_values(env_file_path)
        # add prefix to env vars to avoid name collisions, e.g. PFUND_LIVE_BYBIT_API_KEY
        return {f"{self._env_var_prefix}{k}": v for k, v in raw_env_vars.items()}

    def _parse_data_range(
        self, data_range: str | Resolution | DataRangeDict | Literal["ytd"]
    ) -> tuple[datetime.date, datetime.date]:
        from pfeed.utils.temporal import parse_date_range

        is_data_range_dict = isinstance(data_range, dict)
        return parse_date_range(
            start_date=data_range["start_date"] if is_data_range_dict else "",
            end_date=data_range.get("end_date", "") if is_data_range_dict else "",
            rollback_period=data_range if not is_data_range_dict else "",
        )

    def _load_settings(self) -> BaseEngineSettings:
        """Load settings from settings.toml"""
        from pfund_kit.utils import toml
        from pfund.engines.settings.backtest_engine_settings import (
            BacktestEngineSettings,
        )
        from pfund.engines.settings.sandbox_engine_settings import SandboxEngineSettings
        from pfund.engines.settings.trade_engine_settings import TradeEngineSettings

        settings_file_path = self.pfund_config.get_settings_file_path(self.name)

        if self.env == Environment.BACKTEST:
            EngineSettings = BacktestEngineSettings
        elif self.env == Environment.SANDBOX:
            EngineSettings = SandboxEngineSettings
        elif self.env in [Environment.PAPER, Environment.LIVE]:
            EngineSettings = TradeEngineSettings
        else:
            raise ValueError(f"Unsupported environment: {self.env}")

        if settings_file_path.exists():
            settings_toml = toml.load(settings_file_path)
            env_settings = settings_toml.get(self.env, {})
            settings = EngineSettings(
                **{
                    k: v
                    for k, v in env_settings.items()
                    if k in EngineSettings.model_fields
                }
            )
        else:
            settings = EngineSettings()
        # Always write back — this adds new fields with defaults and drops removed fields automatically
        self._save_settings(settings)
        return settings

    def _save_settings(self, settings: BaseEngineSettings):
        """saves current settings to settings.toml"""
        from pfund_kit.utils import toml

        settings_file_path = self.pfund_config.get_settings_file_path(self.name)
        data = {self.env: settings.model_dump()}
        toml.dump(data, settings_file_path, mode="update", auto_inline=True)

    def get_env(self, key: str) -> str | None:
        """Get env var by key. Automatically adds prefix if not present.

        Examples:
            ctx.get_env('BYBIT_API_KEY')  # looks up PFUND_LIVE_BYBIT_API_KEY
            ctx.get_env('PFUND_LIVE_BYBIT_API_KEY')  # also works
        """
        if not key.startswith(self._env_var_prefix):
            key = f"{self._env_var_prefix}{key}"
        return self._env_vars[key]
