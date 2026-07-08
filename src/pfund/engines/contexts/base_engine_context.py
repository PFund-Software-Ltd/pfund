# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Generic, TypeVar, cast

if TYPE_CHECKING:
    import datetime

    from pfund.engines.base_engine import DataRangeDict
    from pfund.engines.settings.base_engine_settings import BaseEngineSettings
    from pfund.config import PFundConfig
    from pfund.engines.component_registry import RegistryProxy

from pfeed.enums import DataTool

from pfund.config import get_config, get_logging_config
from pfund.datas.resolution import Resolution
from pfund.engines.component_registry import ComponentRegistry
from pfund.enums import Environment, RunMode


SettingsT = TypeVar("SettingsT", bound="BaseEngineSettings")


class BaseEngineContext(Generic[SettingsT]):
    DEFAULT_PROJECT_NAME = "default_project"
    DEFAULT_RUN_NAME = "default_run"

    def __init__(
        self,
        env: Environment | str,
        name: str,
        data_range: str | Resolution | DataRangeDict | tuple[str, str] | Literal["ytd"],
        settings: SettingsT | None = None,
    ):
        import pfeed as pe

        self.env = Environment[env.upper()]
        self.name = name
        self.run_mode = self._detect_run_mode()
        self.project_name = self.DEFAULT_PROJECT_NAME
        self.run_name = self.DEFAULT_RUN_NAME
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

    def _resolve_settings(self, settings: SettingsT | None) -> SettingsT:
        if settings is None:
            return self._load_settings()
        if settings.persist:
            self._save_settings(settings)
        return settings

    def _parse_data_range(
        self,
        data_range: str | Resolution | DataRangeDict | tuple[str, str] | Literal["ytd"],
    ) -> tuple[datetime.date, datetime.date]:
        from pfeed.utils.temporal import parse_date_range

        # normalize a (start_date, end_date) tuple into a DataRangeDict
        if isinstance(data_range, tuple):
            start_date, end_date = data_range
            data_range = {"start_date": start_date, "end_date": end_date}

        is_data_range_dict = isinstance(data_range, dict)
        return parse_date_range(
            start_date=data_range["start_date"] if is_data_range_dict else "",
            end_date=data_range.get("end_date", "") if is_data_range_dict else "",
            rollback_period=data_range if not is_data_range_dict else "",
        )

    def _load_settings(self) -> SettingsT:
        """Load settings from settings.toml"""
        from pfund_kit.utils import toml

        settings_file_path = self.pfund_config.get_settings_file_path(self.name)

        if self.env == Environment.BACKTEST:
            from pfund.engines.settings.backtest_engine_settings import (
                BacktestEngineSettings,
            )

            EngineSettings = BacktestEngineSettings
        elif self.env == Environment.SANDBOX:
            from pfund.engines.settings.sandbox_engine_settings import (
                SandboxEngineSettings,
            )

            EngineSettings = SandboxEngineSettings
        elif self.env in [Environment.PAPER, Environment.LIVE]:
            from pfund.engines.settings.trade_engine_settings import TradeEngineSettings

            EngineSettings = TradeEngineSettings
        else:
            raise ValueError(f"Unsupported environment: {self.env}")

        if settings_file_path.exists():
            settings_toml = toml.load(settings_file_path)
            env_settings = settings_toml.get(self.env, {})
            settings = cast(
                "SettingsT",
                EngineSettings(
                    **{
                        k: v
                        for k, v in env_settings.items()
                        if k in EngineSettings.model_fields
                    }
                ),
            )
        else:
            settings = cast("SettingsT", EngineSettings())
        # Always write back — this adds new fields with defaults and drops removed fields automatically
        self._save_settings(settings)
        return settings

    def _save_settings(self, settings: SettingsT):
        """saves current settings to settings.toml"""
        from pfund_kit.utils import toml

        settings_file_path = self.pfund_config.get_settings_file_path(self.name)
        data = {self.env: settings.model_dump()}
        toml.dump(data, settings_file_path, mode="update", auto_inline=True)

    def set_project_name(self, name: str):
        self.project_name = name

    def set_run_name(self, name: str):
        self.run_name = name
