from pydantic import BaseModel, model_validator


class ExternalListeners(BaseModel):
    notebooks: list[str] | bool = False
    dashboards: list[str] | bool = False
    monitor: bool = False
    recorder: bool = False
    profiler: bool = False

    # TODO: add checking on the notebooks+dashboards names
    @model_validator(mode='after')
    def set_default_notebooks_and_dashboards(self):
        # 'pfund_official/default_dashboard'
        if self.notebooks is True:
            self.notebooks = ['pfund_official/default_notebook']
        if self.dashboards is True:
            self.dashboards = ['pfund_official/default_dashboard']
        return self