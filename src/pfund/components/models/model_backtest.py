from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast, override

if TYPE_CHECKING:
    from collections.abc import Generator

    from pfeed.dataflow.result import RunResult

    from pfund._pfeed_source.component.feed import PFundComponentFeed
    from pfund.typing import ModelT

from contextlib import contextmanager

from pfund.backtest.backtest_mixin import BacktestMixin
from pfund.enums import ArtifactType


def BacktestModel(model: ModelT) -> ModelT:
    """Turns `model` into its backtesting version in place, keeping its state."""
    # already adopted, e.g. the same model instance shared by two parents
    if isinstance(model, BacktestMixin):
        return model
    Model = type(model)

    class _BacktestModel(BacktestMixin, Model):
        _is_training: bool = False
        _checkpoint_artifact: bytes | None = None

        def is_training(self) -> bool:
            return self._is_training

        @override
        def _materialize(self):
            super()._materialize()
            try:
                self.load()
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
            self._checkpoint_artifact = self._encode_checkpoint(checkpoint)
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
            result = cast(
                "PFundComponentFeed",
                self.store._feed.retrieve(
                    artifact_type=ArtifactType.checkpoint,
                    storage_config=self.store.storage_config,
                    checkpoint_step=step,
                ),
            ).run()
            data = result.data
            if data is None:
                raise FileNotFoundError(
                    f"no checkpoint found for '{self.name}' at step {step} in "
                    + f"{self.store.storage_config.storage} storage"
                )
            if not isinstance(data, bytes):
                raise TypeError(
                    f"checkpoint for '{self.name}' at step {step} returned "
                    + f"{type(data).__name__}, expected bytes"
                )
            return self._decode_checkpoint(data)

        # TODO
        def save_checkpoint(self, checkpoint: dict[str, Any], step: int) -> RunResult:
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
                return cast(
                    "PFundComponentFeed",
                    self.store._feed.download(
                        artifact_type=ArtifactType.checkpoint,
                        storage_config=self.store.storage_config,
                        checkpoint_step=step,
                    ),
                ).run()

    _BacktestModel.__name__ = Model.__name__
    _BacktestModel.__qualname__ = Model.__qualname__
    _BacktestModel.__wrapped__ = Model

    return _BacktestModel._adopt(model)
