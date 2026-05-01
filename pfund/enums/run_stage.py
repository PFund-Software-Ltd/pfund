from enum import StrEnum


class RunStage(StrEnum):
    EXPERIMENT = 'EXPERIMENT'
    REFINEMENT = 'REFINEMENT'
    DEPLOYMENT = 'DEPLOYMENT'