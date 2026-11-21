# ty: ignore[incompatible-attribute-override,incompatible-mutable-override]
from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal

if TYPE_CHECKING:
    from pfund._pfeed_source.source import PFund  # noqa: F401, used as a string forward reference in the class bases

from pfeed.base.requests.base_request import BaseRequest

from pfund._pfeed_source.base.feed import PFundBaseFeed
from pfund._pfeed_source.data_category import PFundDataCategory
from pfund._pfeed_source.engine.data_model import PFundEngineDataModel


# TODO: add StreamingMixin
class PFundEngineFeed[PipelineT: bool = Literal[False]](
    PFundBaseFeed[
        BaseRequest,
        PFundEngineDataModel,
        PipelineT,
    ]
):
    DataModel: ClassVar[type[PFundEngineDataModel]] = PFundEngineDataModel
    data_domain: ClassVar[PFundDataCategory] = PFundDataCategory.ENGINE_DATA

    @classmethod
    def create_data_model(cls, *args: Any, **kwargs: Any) -> PFundEngineDataModel:
        raise NotImplementedError(
            f"{cls.__name__}.create_data_model() is not implemented yet"
        )
