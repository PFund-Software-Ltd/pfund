from pydantic import BaseModel, Field, field_validator


'''
settings: BacktestEngineSettings = {
        'commit_to_git': False,
        'retention_period': '7d',
    }
'''
class BacktestEngineSettings(BaseModel):
    pass
