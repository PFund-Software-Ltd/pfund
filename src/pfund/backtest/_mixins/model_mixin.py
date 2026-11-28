from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast, override

if TYPE_CHECKING:
    from pfund.components.models.model_base import BaseModel

import datetime

from pydantic import BaseModel as PydanticModel

from pfund.backtest._mixins.backtest_mixin import BacktestMixin
from pfund.backtest.cv.dataset_splits import FoldNum
from pfund.enums import ArtifactType


class TrainedOn(PydanticModel):
    """The rows a model was trained on, stamped on its model and checkpoint artifacts.

    The engine's data range alone doesn't say it: a model trained on the train set of
    T1-T100 only saw e.g. T1-T70, see BacktestModelMixin._get_trained_on().
    """

    # the split part in use, e.g. "train", set by engine.use_split() or the default 100% train
    part: str
    # the cross-validation fold, None under holdout
    fold: FoldNum | None = None
    # the grid the split was cut on, e.g. "1d"
    split_resolution: str
    # [start, end) of each contiguous run of the rows, a cross-validation fold may have gaps
    intervals: list[tuple[datetime.datetime, datetime.datetime]]


class BacktestModelMixin(BacktestMixin):
    """Backtest behavior specific to models, including training provenance and checkpoints.

    Adopted onto a model via ``BacktestModelMixin._adopt(model)``.
    """

    @override
    def _materialize(self):
        # Child models may need their weights to generate signals during build.
        if not self._is_backtestee:
            try:
                self.load()
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"No trained model found for model '{self.name}'. "
                ) from None
        super()._materialize()

    def load(self) -> None:
        """Loads saved weights and checks for overlap with the backtest data range."""
        cast("BaseModel", super()).load()
        self._warn_in_sample()

    def _get_in_sample_intervals(
        self,
    ) -> list[tuple[datetime.datetime, datetime.datetime]] | None:
        """The parts of this run's data range the loaded model was trained on, ``[start, end)``.

        Its predictions there are in-sample, i.e. look better than they would on unseen data.
        An empty list if they don't overlap, None if what the model was trained on is unknown.
        """
        trained_on = cast("BaseModel", self).trained_on
        if trained_on is None:
            return None
        context = self.context
        assert context.data_start is not None and context.data_end is not None
        run_start = datetime.datetime.combine(context.data_start, datetime.time.min)
        run_end = datetime.datetime.combine(
            context.data_end + datetime.timedelta(days=1), datetime.time.min
        )
        overlaps = [
            (max(start, run_start), min(end, run_end))
            for start, end in trained_on.intervals
        ]
        return [(start, end) for start, end in overlaps if start < end]

    def _warn_in_sample(self) -> None:
        """Warns if the loaded model predicts on rows it was trained on, see _get_in_sample_intervals()."""
        from pfund_kit.style import RichColor, TextStyle

        in_sample = self._get_in_sample_intervals()
        if in_sample == []:
            return
        if in_sample is None:
            reason = "what it was trained on wasn't recorded, it may overlap this run's data range"
        else:
            ranges = ", ".join(
                f"{start.isoformat()} -> {end.isoformat()}" for start, end in in_sample
            )
            reason = f"it was trained on {ranges} of this run's data range"
        self.logger.warning(
            f"'{self.name}' predictions are in-sample where it was trained: {reason}. "
            + "Its results there look better than on unseen data, "
            + "evaluate it out-of-sample, e.g. under engine.use_split('test')",
            style=TextStyle.BOLD + RichColor.YELLOW,
        )

    def save(self) -> None:
        """Persists the trained model, stamped with what it was trained on, see _get_trained_on()."""
        model = cast("BaseModel", self)
        # raises if what the model was trained on is unknown, before anything is written
        self._trained_on = self._get_trained_on()
        self._feed.save_artifact(model, ArtifactType.model, artifact=model._encode())

    def _get_trained_on(self) -> TrainedOn:
        """What the model was trained on (BaseModel.trained_on), from the split set by engine.use_split(),
        set before it's stamped on a model or checkpoint artifact.

        pfund doesn't see what fit() receives, so with dataset_splits beyond 100% train on the engine,
        the model must be trained and saved under engine.use_split(), with engine.apply_split() for its target etc.
        With 100% train (the default), the train set is in use without use_split().

        Raises:
            RuntimeError: no split is in use.
            ValueError: the split in use is the val or test set.
        """
        dataset_splits = self.dataset_splits
        part, fold = self.active_split
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
        self._trained_on = self._get_trained_on()
        self._feed.save_artifact(
            cast("BaseModel", self),
            ArtifactType.checkpoint,
            checkpoint_step=step,
            # _encode_checkpoint is optional, defined only by models that support checkpoints (e.g. PyTorchModel)
            artifact=cast("Any", self)._encode_checkpoint(checkpoint),
        )
