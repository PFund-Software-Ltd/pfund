from pydantic import Field

from pfund.settings.base_engine_settings import BaseEngineSettings


'''
settings: BacktestEngineSettings = {
        'commit_to_git': False,
        'retention_period': '7d',
    }
'''
class BacktestEngineSettings(BaseEngineSettings):
    retention_period: int = Field(default=7, ge=1)
    commit_to_git: bool = Field(default=False)
