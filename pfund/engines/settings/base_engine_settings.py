from pydantic import BaseModel, Field, ConfigDict, model_validator


class BaseEngineSettings(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid', frozen=True)

    df_min_rows: int = Field(default=1_000, ge=1)
    df_max_rows: int = Field(default=3_000, ge=1)
    
    @model_validator(mode="after")
    def check_df_rows(self):
        if self.df_max_rows < self.df_min_rows:
            raise ValueError(f"df_max_rows ({self.df_max_rows}) must be greater than or equal to df_min_rows ({self.df_min_rows})")
        return self
