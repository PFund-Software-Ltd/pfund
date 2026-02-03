from __future__ import annotations
from typing import TYPE_CHECKING, Literal, TypedDict
if TYPE_CHECKING:
    import datetime

from pfund.enums import Environment
from pfund.datas.resolution import Resolution
from pfund.settings import TradeEngineSettings, BacktestEngineSettings


class DataRangeDict(TypedDict, total=False):
    start_date: str
    end_date: str


class EngineContext:
    def __init__(
        self,
        env: Environment,
        data_range: str | Resolution | DataRangeDict | Literal['ytd'],
    ):
        self.env = env
        self._env_var_prefix = f'PFUND_{env.upper()}_'
        self._env_vars = self._load_env_vars()
        self.data_start, self.data_end = self._parse_data_range(data_range)
        self.settings = self._load_settings()
    
    def _load_env_vars(self) -> dict[str, str]:
        from dotenv import find_dotenv, dotenv_values
        # load env vars manually to avoid loading into os.environ
        # NOTE: this allows multiple engines with different envs in the same process
        env_filename = f'.env.{self.env.lower()}'
        env_file_path = find_dotenv(filename=env_filename, usecwd=True, raise_error_if_not_found=False)
        raw_env_vars = dotenv_values(env_file_path)
        # add prefix to env vars to avoid name collisions, e.g. PFUND_LIVE_BYBIT_API_KEY
        return {f'{self._env_var_prefix}{k}': v for k, v in raw_env_vars.items()}
    
    def _parse_data_range(self, data_range: str | Resolution | DataRangeDict | Literal['ytd']) -> tuple[datetime.date, datetime.date]:
        from pfeed.utils import parse_date_range
        is_data_range_dict = isinstance(data_range, dict)
        return parse_date_range(
            start_date=data_range['start_date'] if is_data_range_dict else '',
            end_date=data_range.get('end_date', '') if is_data_range_dict else '',
            rollback_period=data_range if not is_data_range_dict else '',
        )
    
    def _load_settings(self) -> TradeEngineSettings | BacktestEngineSettings:
        '''Load settings from settings.toml'''
        from pfund_kit.utils import toml
        from pfund import get_config
        
        config = get_config()

        EngineSettings = BacktestEngineSettings if self.env == Environment.BACKTEST else TradeEngineSettings
        settings_file_path = config.settings_file_path
        if settings_file_path.exists():
            settings_toml = toml.load(settings_file_path)
            env_settings = settings_toml.get(self.env, {})
            settings = EngineSettings(
                **{k: v for k, v in env_settings.items() if k in EngineSettings.model_fields}
            )
            if self.env not in settings_toml:
                data = {self.env: settings.model_dump()}
                toml.dump(data, settings_file_path, mode='update', auto_inline=True)
        else:
            settings = EngineSettings()
            data = {self.env: settings.model_dump()}
            toml.dump(data, settings_file_path, mode='overwrite', auto_inline=True)
        return settings

    def get_env(self, key: str) -> str | None:
        """Get env var by key. Automatically adds prefix if not present.

        Examples:
            ctx.get_env('BYBIT_API_KEY')  # looks up PFUND_LIVE_BYBIT_API_KEY
            ctx.get_env('PFUND_LIVE_BYBIT_API_KEY')  # also works
        """
        if not key.startswith(self._env_var_prefix):
            key = f'{self._env_var_prefix}{key}'
        return self._env_vars[key]
