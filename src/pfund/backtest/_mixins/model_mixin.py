from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast, override

if TYPE_CHECKING:
    from pfund._pfeed_source.component.data_handler import TrainedOn
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

    def _get_trained_on(self) -> TrainedOn | None:
        """The split set by engine.use_split() when the model is saved, see BaseModel._get_trained_on().

        pfund doesn't see what fit() receives, so with dataset_splits on the engine the model must be
        trained and saved under engine.use_split(), with engine.apply_split() for its target etc.

        Raises:
            RuntimeError: the engine has dataset_splits but no split is in use, i.e. it's unknown what
                the model was trained on, and the whole data range would include the test set.
            ValueError: the split in use is the val or test set.
        """
        from pfund._pfeed_source.component.data_handler import TrainedOn

        dataset_splits = self.dataset_splits
        context = self.context
        if dataset_splits is None:
            assert context.data_start is not None and context.data_end is not None
            return TrainedOn.whole_range(context.data_start, context.data_end)
        if context.active_split is None:
            raise RuntimeError(
                f"'{self.name}' is saved with dataset_splits on the engine but no split in use, "
                + "so what it was trained on is unknown, and the whole data range includes the test set. "
                + "Train and save it under the split instead:\n"
                + "    with engine.use_split('train'):\n"
                + "        model.fit(model.X, engine.apply_split(y))\n"
                + "        model.save()"
            )
        part, fold = context.active_split
        if part in ("val", "test"):
            raise ValueError(
                f"'{self.name}' is saved under engine.use_split({part!r}), "
                + f"a model must not be trained on the {part} set, use 'train' or 'trainval'"
            )
        return TrainedOn(
            part=part,
            fold=fold,
            split_resolution=repr(dataset_splits.split_resolution),
            intervals=dataset_splits.intervals(part, fold),
        )

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
