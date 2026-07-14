from enum import StrEnum


class ModelComponentType(StrEnum):
    model = "model"
    feature = "feature"


class ComponentType(StrEnum):
    strategy = "strategy"
    model = ModelComponentType.model
    feature = ModelComponentType.feature

    def to_plural(self) -> str:
        if self == ComponentType.strategy:
            return "strategies"
        else:
            return self.value + "s"
