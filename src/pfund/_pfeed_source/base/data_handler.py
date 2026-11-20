from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pfeed.io.blob_io import BlobIO
    from pfeed.io.table_io import TableIO
    from pfund_blocks.enums import Environment

import logging

from pfeed.base.data_handler import BaseDataHandler, BaseDataMetadata

from pfund._pfeed_source.base.data_model import PFundBaseDataModel


class PFundBaseDataHandler[
    DataModelT: PFundBaseDataModel,
    MetadataT: BaseDataMetadata,
](BaseDataHandler[DataModelT, MetadataT]):
    @classmethod
    def clear_run(
        cls,
        env: Environment,
        project: str,
        ios: list[TableIO | BlobIO],
        run: str = PFundBaseDataModel.DEFAULT_RUN_NAME,
    ) -> None:
        """Deletes everything a run stored in each IO, used to reset default_run/ before a run without mtflow.

        Only default_run can be cleared, so an mtflow run (e.g. run_001) is never deleted by mistake.

        Args:
            env: the run's environment.
            project: the run's project.
            ios: every IO the run stored in, e.g. the context's datalake and the artifacts' BlobIO.
            run: the run to clear, must be default_run.
        """
        if run != PFundBaseDataModel.DEFAULT_RUN_NAME:
            raise ValueError(
                f"only {PFundBaseDataModel.DEFAULT_RUN_NAME} can be cleared, got {run}"
            )
        # same namespace as every pfund DatasetKey, see PFundComponentDataHandler._create_dataset_key()
        namespace = {"env": str(env), "project": project.lower(), "run": run}
        for io in ios:
            io.delete_namespace(namespace)
        logging.getLogger("pfund").debug(f"cleared run {namespace}")

    # FIXME: moved as-is from BaseEngine._clear_run_path; DataStorage/FileBasedStorage/StorageConfig
    # are pfeed's old storage API, which pfeed's DatasetKey-based IO has replaced


#     def _clear_run_path(self, *, confirm: bool = False) -> None:
#         from pathlib import Path
#
#         import pyarrow.fs as pa_fs
#
#         storage_config = self.context.datalake_storage_config
#         Storage = DataStorage[storage_config.storage].storage_class
#         storage = Storage.from_storage_config(storage_config)
#
#         if not isinstance(storage, FileBasedStorage):
#             raise TypeError(f"Cannot clear a filesystem run path using {storage.name}")
#
#         run_path = self._create_run_path(
#             data_path=Path(cast("FilePath", storage.data_path).schemeless),
#             env=self.env,
#             project=self.context.project_name,
#             run=self.context.run_name,
#         )
#
#         filesystem = storage.get_filesystem()
#         file_info = filesystem.get_file_info(str(run_path))
#
#         if file_info.type == pa_fs.FileType.NotFound:
#             return
#
#         if file_info.type != pa_fs.FileType.Directory:
#             raise RuntimeError(f"Run path is not a directory: {run_path}")
#
#         if confirm:
#             self._confirm_clear_run_path(run_path)
#
#         filesystem.delete_dir(str(run_path))
#         self._logger.debug(f"cleared existing run path: {run_path}")
#
#     def _confirm_clear_run_path(self, run_path: Path) -> None:
#         import signal
#
#         from rich.markup import escape
#         from rich.prompt import Prompt
#
#         class _OverwriteCancelled(Exception):
#             pass
#
#         def _handle_sigint(_signum, _frame):
#             raise _OverwriteCancelled
#
#         previous_sigint_handler = signal.getsignal(signal.SIGINT)
#         signal.signal(signal.SIGINT, _handle_sigint)
#         try:
#             choice = Prompt.ask(
#                 (
#                     "\n[bold red]WARNING:[/] engine's run(overwrite=True) will permanently delete "
#                     "the whole run folder:\n\n"
#                     f"  {escape(str(run_path))}\n\n"
#                     "This includes all component artifacts (e.g. models).\n\n"
#                     "\\[y] Clear this time\n"
#                     "\\[n] Cancel\n"
#                     "\\[d] Clear and disable this warning in settings.toml"
#                 ),
#                 choices=["y", "n", "d"],
#                 case_sensitive=False,
#                 default="n",
#                 show_choices=False,
#             ).lower()
#         except _OverwriteCancelled:
#             choice = "n"
#         finally:
#             signal.signal(signal.SIGINT, previous_sigint_handler)
#
#         if choice == "n":
#             print("\nRun cancelled; the existing run folder was not cleared")
#             raise SystemExit(0)
#
#         if choice == "d":
#             self.settings.warn_overwrite = False
#             self.settings.save(self.name, self.env)
