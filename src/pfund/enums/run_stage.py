from enum import StrEnum


class RunStage(StrEnum):
    experiment = "experiment"
    refinement = "refinement"
    deployment = "deployment"

    def to_plural(self) -> str:
        return self.value + "s"
