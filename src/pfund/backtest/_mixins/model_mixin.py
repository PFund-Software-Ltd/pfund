from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast, override

if TYPE_CHECKING:
    from pfund.components.models.model_base import BaseModel

from pfund.backtest._mixins.backtest_mixin import BacktestMixin
from pfund.enums import ArtifactType


class BacktestModelMixin(BacktestMixin):
    """Backtest behavior specific to models, e.g. training detection and checkpoints.

    Adopted onto a model via ``BacktestModelMixin._adopt(model)``.
    """

    _is_training: bool = False

    def is_training(self) -> bool:
        return self._is_training

    @override
    def _materialize(self):
        # load first: super()._materialize() may predict on the fly, which needs the trained model
        try:
            # the mixin only ever runs as part of a model
            cast("BaseModel", self).load()
        except FileNotFoundError:
            from pfund_kit.style import RichColor, TextStyle

            # only the backtestee can be trained, a child model must already be trained
            if not self._is_backtestee:
                raise FileNotFoundError(
                    f"No trained model found for model '{self.name}'. "
                ) from None
            else:
                self.logger.warning(
                    f"No trained model found for '{self.name}' — treating this run as a training session. "
                    + "Fit your model, then call save() so future runs can load it and generate predictions.",
                    style=TextStyle.BOLD + RichColor.YELLOW,
                )
                self._is_training = True
        super()._materialize()

    def load_checkpoint(self, step: int) -> dict[str, Any]:
        """Read back the checkpoint written by save_checkpoint() at `step`. Returns the user's
        own dict (weights, optimizer, epoch, ...) so they can resume their loop.
        """
        if not hasattr(self, "_decode_checkpoint"):
            raise NotImplementedError(
                "load_checkpoint is not implemented for this model"
            )
        data, metadata = self._feed.load_artifact(
            cast("BaseModel", self),
            ArtifactType.checkpoint,
            checkpoint_step=step,
            include_metadata=True,
        )
        if data is None:
            raise FileNotFoundError(
                f"no checkpoint found for '{self.name}' at step {step}"
            )
        if not isinstance(data, bytes):
            raise TypeError(
                f"checkpoint for '{self.name}' at step {step} returned "
                + f"{type(data).__name__}, expected bytes"
            )
        # warn only, like BaseModel.load(): the user decides whether to resume from it
        if reason := self._get_stale_reason(metadata):
            self.logger.warning(
                f"'{self.name}' checkpoint at step {step} is stale: {reason}"
            )
        return self._decode_checkpoint(data)

    def save_checkpoint(self, checkpoint: dict[str, Any], step: int) -> None:
        """Persist a training checkpoint the user built (weights, optimizer, epoch,
        ... — their dict, their content). Written under the same model storage as
        save(), but keyed by `step` so checkpoints coexist without clobbering the
        final model or each other.
        """
        if not hasattr(self, "_encode_checkpoint"):
            raise NotImplementedError(
                "save_checkpoint is not implemented for this model"
            )
        self._feed.save_artifact(
            cast("BaseModel", self),
            ArtifactType.checkpoint,
            checkpoint_step=step,
            # _encode_checkpoint is optional, defined only by models that support checkpoints (e.g. PyTorchModel)
            artifact=cast("Any", self)._encode_checkpoint(checkpoint),
        )
