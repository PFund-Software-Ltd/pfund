from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast, override

if TYPE_CHECKING:
    from collections.abc import Generator

    from pfund.components.models.model_base import BaseModel

from contextlib import contextmanager

from pfund.backtest._mixins.backtest_mixin import BacktestMixin
from pfund.enums import ArtifactType


class BacktestModelMixin(BacktestMixin):
    """Backtest behavior specific to models, e.g. training detection and checkpoints.

    Adopted onto a model via ``BacktestModelMixin._adopt(model)``.
    """

    _is_training: bool = False
    _checkpoint_artifact: bytes | None = None

    def is_training(self) -> bool:
        return self._is_training

    @override
    def _materialize(self):
        super()._materialize()
        try:
            # the mixin only ever runs as part of a model
            cast("BaseModel", self).load()
        except FileNotFoundError:
            from pfund_kit.style import RichColor, TextStyle, cprint

            cprint(
                f"No trained model found for '{self.name}' — treating this run as a training session. "
                + "Fit your model, then call save() so future runs can load it and generate predictions.",
                style=TextStyle.BOLD + RichColor.YELLOW,
            )
            self._is_training = True

    @contextmanager
    def _stage_checkpoint_artifact(
        self,
        checkpoint: dict[str, Any],
    ) -> Generator[None]:
        """Temporarily expose a user-built checkpoint as component state.

        pfeed's ComponentFeed treats a pfund component as a data source:
        ``download()`` should pull an artifact from the bound component
        instead of receiving the artifact data as an argument.
        Unlike model weights or the trading dataframe, a training checkpoint originates in user code
        and is not normally retained by the model.
        Staging it for the duration of ``download().run()`` lets pfeed extract
        it through ``checkpoint_artifact`` while preserving that pull-based contract.
        The ``finally`` block ensures the temporary state never outlives the run.
        """
        if self._checkpoint_artifact is not None:
            raise RuntimeError("another checkpoint is already staged")
        # _encode_checkpoint is optional, defined only by models that support checkpoints
        # (e.g. PyTorchModel), save_checkpoint() checks it with hasattr before staging
        self._checkpoint_artifact = cast("Any", self)._encode_checkpoint(checkpoint)
        try:
            yield
        finally:
            self._checkpoint_artifact = None

    def load_checkpoint(self, step: int) -> dict[str, Any]:
        """Read back the checkpoint written by save_checkpoint() at `step`. Restores
        the framework essentials (signal_cols) into the model and returns the user's
        own dict (weights, optimizer, epoch, ...) so they can resume their loop.
        """
        if not hasattr(self, "_decode_checkpoint"):
            raise NotImplementedError(
                "load_checkpoint is not implemented for this model"
            )
        data = self.store._feed.load_artifact(
            cast("BaseModel", self), ArtifactType.checkpoint, checkpoint_step=step
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
        return self._decode_checkpoint(data)

    # TODO
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
        # TODO: push the checkpoint instead of staging it on the component.
        # The checkpoint is user input, not component state; it's staged only because download() pulls artifacts from the component.
        # e.g. download(artifact_type=checkpoint, checkpoint_step=step, data=self._encode_checkpoint(checkpoint)),
        # and PFundComponentFeed._download_impl returns request.data for ArtifactType.checkpoint.
        # removes _checkpoint_artifact, _stage_checkpoint_artifact() and the "No checkpoint is staged" branch,
        # _encode_checkpoint/_decode_checkpoint stay on the model.
        with self._stage_checkpoint_artifact(checkpoint):
            self.store._feed.save_artifact(
                cast("BaseModel", self), ArtifactType.checkpoint, checkpoint_step=step
            )
