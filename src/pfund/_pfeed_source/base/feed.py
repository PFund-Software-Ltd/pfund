from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, override

from collections.abc import Callable

if TYPE_CHECKING:
    from pfund._pfeed_source.source import PFund

from pfeed.base.feed import BaseFeed

from pfund._pfeed_source.base.data_model import PFundBaseDataModel
from pfund._pfeed_source.base.requests.base_request import PFundBaseRequest


class PFundBaseFeed[
    RequestT: PFundBaseRequest,
    DataModelT: PFundBaseDataModel,
    PipelineT: bool = Literal[False],
](
    BaseFeed[
        "PFund[PipelineT]",
        RequestT,
        DataModelT,
        PipelineT,
    ]
):
    @override
    def __init__(
        self,
        data_source: PFund[PipelineT],
        pipeline_mode: bool = False,
        num_workers: int | None = None,
    ):
        # pfund's feeds are eager, one artifact/state per call, there is nothing to queue and run later
        if pipeline_mode:
            raise ValueError(
                f"{type(self).__name__} doesn't support pipeline mode, pipeline_mode must be False"
            )
        super().__init__(
            data_source, pipeline_mode=pipeline_mode, num_workers=num_workers
        )

    @override
    def set_num_workers(self, num_workers: int | None) -> None:
        # using ray is not allowed, component is included in request and it can't be pickled
        if num_workers is not None:
            raise ValueError(
                f"{self.name}'s component feed doesn't support Ray, num_workers must be None"
            )

    def _get_default_transformations(
        self, request: RequestT
    ) -> list[Callable[..., Any]]:
        return []
