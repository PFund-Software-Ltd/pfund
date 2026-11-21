from pfeed.base.requests.base_request import BaseRequest
from pfund_blocks.enums.env import Environment


class PFundFeedBaseRequest(BaseRequest):
    """Base request of pfund's feeds, identifies the engine run the data belongs to.

    Attributes:
        env: environment of the run, e.g. BACKTEST.
        project: project name of the run
        run: name of the run
    """

    env: Environment
    # NOTE: when it's project/run_name is none, default values in PFundBaseDataModel will be used
    project: str | None = None
    run: str | None = None

    def should_clean_data(self) -> bool:
        # artifacts, engine states are returned as stored
        return False
