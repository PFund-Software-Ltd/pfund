from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from pfund._pfeed_source.component.data_handler import PFundComponentDataMetadata
    from pfund.components._mixins.component_mixin import ComponentMixin

import hashlib
from functools import cached_property
from pathlib import Path

from pfund_kit.utils.json import stable_dumps

from pfund.utils.decorators import ray_method


def _normalize_source(source: str) -> str:
    """The class source without what can't change behavior: comments, formatting and docstrings.

    Dumps the source's AST, which keeps no comments, whitespace or line numbers, with every docstring removed.
    Falls back to the raw source if it doesn't parse on its own (e.g. a nested class whose
    multi-line string breaks the dedent), so a change is never missed, only possibly over-reported.
    NOTE: ast.dump()'s output can change across Python versions, so upgrading Python can mark artifacts stale once.
    """
    import ast
    import textwrap

    try:
        tree = ast.parse(textwrap.dedent(source))
    except SyntaxError:
        return source
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        # same rule as ast.get_docstring(): a string literal as the first statement of the body
        first = node.body[0] if node.body else None
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            node.body = node.body[1:]
    return ast.dump(tree)


class ArtifactMixin:
    """What a component stamps on its artifacts and checks them against.

    Every artifact (trading df, model, checkpoint, source) is stamped with the component's setup and
    source hashes when written, and compared with them on load, see _get_stale_reason().
    """

    def _get_source_classes(self) -> list[type]:
        """The classes whose source is the component's code, hashed by _source_sha256.

        The component's MRO, so editing a user-defined base class (e.g. MyBase subclassed by Alpha) counts too.
        Overridden by BaseModel to add the underlying model's classes, e.g. a torch nn.Module.
        """
        return list(type(self).__mro__)

    # Frozen at first use like _setup_sha256, every artifact of the run is stamped with the same hash
    @cached_property
    def _source_sha256(self) -> str:
        """Hashes the source of every user-written class in _get_source_classes().

        Each artifact is stamped with it when written, and compared on load, see _setup_sha256.
        Only the classes' source, not their whole files: editing a script that defines a component
        (e.g. uncommenting a line that runs it) must not mark the component's artifacts as stale.
        Comments, formatting and docstrings inside a class don't count either, see _normalize_source().
        Classes from pfund, the standard library and installed packages are skipped.
        Module-level helpers a class uses are not covered.
        """
        import inspect
        import sysconfig

        excluded_dirs = [
            Path(sysconfig.get_paths()[key]).resolve()
            for key in ("stdlib", "platstdlib", "purelib", "platlib")
        ]
        sources: list[str] = []
        for klass in dict.fromkeys(self._get_source_classes()):
            if klass.__module__.split(".")[0] == "pfund":
                continue
            try:
                source_file = inspect.getsourcefile(klass)
            except TypeError:  # built-in classes, e.g. object
                continue
            if source_file is not None and any(
                Path(source_file).resolve().is_relative_to(d) for d in excluded_dirs
            ):
                continue
            try:
                source = _normalize_source(inspect.getsource(klass))
            # e.g. a class created dynamically, it has no source
            except (OSError, TypeError):
                continue
            if source not in sources:
                sources.append(source)

        digest = hashlib.sha256()
        for source in sources:
            digest.update(source.encode())
        return digest.hexdigest()

    # Frozen at first use, so the component keeps one setup hash for its lifetime: runtime work
    # can mutate objects in the setup (e.g. a PyTorch LazyLinear materializing on its first
    # forward pass), and rehashing it would stamp the component's artifacts with different hashes.
    # NOTE: _build() freezes it before _materialize(), which mutates the datas, see ComponentMixin._build()
    @cached_property
    def _setup_sha256(self) -> str:
        """Hashes the component's setup, see _setup_fields().

        The artifacts are found by the component's name (its dir, unique across the engine, see
        PFundComponentDataModel.component_path), then validated by stamps written on each of them:
        - `_setup_sha256` answers "were these produced by this component as it is set up now?",
          i.e. the same params, config, data and children.
        - `_source_sha256` answers "were these produced by the code that's running now?"
        - the data range answers "were these produced on the data this run uses?", trading df only.
        See _get_stale_reason().
        """
        try:
            payload = stable_dumps(cast("ComponentMixin", self)._setup_fields())
        except TypeError as err:
            # stable_dumps() only allows encodings that are the same in every run,
            # otherwise the same setup would get a new hash each run and its artifacts would always be stale
            raise TypeError(f"{self} setup: {err}") from err
        return hashlib.sha256(payload.encode()).hexdigest()

    def get_artifact_stamp(self) -> dict[str, str]:
        """The setup and source hashes, as a method so it also works through ActorProxy (Ray only forwards methods).

        A parent's setup holds its children's stamps, so a change in a child (its params or its code)
        makes the parent's artifacts stale too.
        """
        return {
            "setup_sha256": self._setup_sha256,
            "source_sha256": self._source_sha256,
        }

    def _get_stale_reason(
        self,
        metadata: PFundComponentDataMetadata | None,
        *,
        check_data_range: bool = False,
    ) -> str | None:
        """Why an artifact stamped with `metadata` doesn't belong to the component as it runs now, None if it does.

        Args:
            metadata: the artifact's per-partition metadata, None if it has none.
            check_data_range: also require the artifact to be written on the engine's data range,
                for a trading df, whose rows depend on it. A trained model doesn't.
        """
        if metadata is None:
            return "it has no stamp"
        if metadata.setup_sha256 != self._setup_sha256:
            return "its setup (params, config, data or child components) has changed"
        if metadata.source_sha256 != self._source_sha256:
            return "its code has changed"
        context = cast("ComponentMixin", self).context
        data_range = (context.data_start, context.data_end)
        if check_data_range and (metadata.data_start, metadata.data_end) != data_range:
            return (
                f"it was saved on data range ({metadata.data_start}, {metadata.data_end}), "
                + f"not on ({data_range[0]}, {data_range[1]})"
            )
        return None
