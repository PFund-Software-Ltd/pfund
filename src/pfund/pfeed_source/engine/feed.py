from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from collections.abc import Callable

    from pfeed.base.request import BaseRequest

    from pfeed_pfund.source import PFund  # noqa: F401, used as a string forward reference in the class bases

from pfeed.base.feed import BaseFeed
from pfeed.enums import DataCategory

from pfeed_pfund.engine.data_model import PFundEngineDataModel


# TODO: get around the data source issue, since technically its data source is the engine.
# this feed should be able to get backtesting data from pfund's BacktestEngine for monitoring and analysis puporse
# some functions require api calls (e.g. get dynamic backtest results) and some do not (e.g. load backtest hisory)
# TODO: need to specify which engine if there are multiple engines running
class PFundEngineFeed(BaseFeed["PFund", "BaseRequest", PFundEngineDataModel]):
    class Requests:  # ty: ignore[invalid-attribute-override]
        """Request class per verb, None = not supported."""

        # TODO: download = pull the live engine's states (passed in the request), see component feed
        download: ClassVar[None] = None
        # TODO: retrieve = read-back twin of download+load
        retrieve: ClassVar[None] = None
        # TODO: stream engine's states, connect to mtflow's ws server
        stream: ClassVar[None] = None

    DataModel: ClassVar[type[PFundEngineDataModel]] = PFundEngineDataModel  # ty: ignore[invalid-mutable-override]
    data_domain: ClassVar[DataCategory] = DataCategory.ENGINE_DATA

    @classmethod
    def create_data_model(cls, *args: Any, **kwargs: Any) -> PFundEngineDataModel:
        raise NotImplementedError(f"{cls.__name__}.create_data_model() is not implemented yet")

    def _get_default_transformations(self, request: BaseRequest, /) -> list[Callable[..., Any]]:
        raise NotImplementedError(f"{type(self).__name__} has no verbs yet")

    def run(self, use_prefect: bool = False, **prefect_kwargs: Any) -> Any:
        raise NotImplementedError(f"{type(self).__name__} has no verbs yet")
