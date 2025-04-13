from enum import StrEnum


class ModelComponentType(StrEnum):
    model = 'model'
    feature = 'feature'
    indicator = 'indicator'


class ComponentType(StrEnum):
    strategy = 'strategy'
    model = ModelComponentType.model
    indicator = ModelComponentType.indicator
    feature = ModelComponentType.feature