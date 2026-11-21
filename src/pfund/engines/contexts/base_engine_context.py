from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    import datetime

    from pfund.config import PFundConfig
    from pfund.engines.base_engine import DataRangeDict
    from pfund.engines.settings.base_engine_settings import BaseEngineSettings

from pfund_blocks.enums.env import Environment
from pfund_blocks.resolution import Resolution

from pfund.config import get_config, get_logging_config
from pfund.enums import RunMode


class BaseEngineContext[SettingsT: BaseEngineSettings]:
    def __init__(
        self,
        *,
        env: Environment | str,
        name: str,
        data_range: str
        | Resolution
        | DataRangeDict
        | tuple[str, str]
        | Literal["ytd"]
        | None,
        settings: SettingsT | None = None,
    ):
        self.env = Environment[env.upper()]
        self.name = name
        # REVIEW: engine has no REMOTE run mode for now
        self.run_mode = RunMode.LOCAL
        self.data_start, self.data_end = self._parse_data_range(data_range)
        # NOTE: config obtained by get_config() inside ray actor could be different from the one in the main thread (e.g. after calling pf.configure())
        # so we create the config object here in the context and treat it as the source of truth
        self.config: PFundConfig = get_config().scoped(self.name)
        self.logging_config: dict[str, Any] = get_logging_config()
        self.settings = self._resolve_settings(settings)

    def _resolve_settings(self, settings: SettingsT | None) -> SettingsT:
        # load settings if it's None
        if settings is None:
            from pfund.engines.settings import ENGINE_SETTINGS

            return cast(
                "SettingsT", ENGINE_SETTINGS[self.env].load(self.name, self.env)
            )
        else:
            if settings.persist:
                settings.save(self.name, self.env)
        return settings

    def _parse_data_range(
        self,
        data_range: str
        | Resolution
        | DataRangeDict
        | tuple[str, str]
        | Literal["ytd"]
        | None,
    ) -> tuple[datetime.date | None, datetime.date | None]:
        from pfeed.utils.temporal import parse_date_range

        start_date = end_date = rollback_period = None

        if data_range is None:
            return None, None
        elif isinstance(data_range, tuple):
            start_date, end_date = data_range
        elif isinstance(data_range, (str, Resolution)):
            rollback_period = data_range
        else:
            start_date, end_date = data_range["start_date"], data_range.get("end_date")

        return parse_date_range(
            start_date=start_date,
            end_date=end_date,
            rollback_period=rollback_period,
        )
