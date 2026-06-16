from __future__ import annotations

import importlib.util
from concurrent.futures import ThreadPoolExecutor
from importlib.resources import files
from pathlib import Path


# Root package holding the bundled, official notebooks: pfund.hub.notebooks.<group>.<name>.py
_NOTEBOOKS_ROOT = "pfund.hub.notebooks"


def _download(name: str, *, local_dir: Path, force: bool) -> str:
    """Copy a single official notebook into a local, user-owned dir.

    Args:
        name: The notebook as "<group>/<notebook>" (the ".py" suffix is optional),
            e.g. "pfund_official/default_backtest". The group is required; a bare
            "default_backtest" raises ValueError.
        local_dir: Directory to copy the notebook into. It keeps its
            "<group>/<notebook>.py" sub-path under here.
        force: If False, skip the copy when the notebook already exists locally
            (preserving local edits). If True, overwrite it with the official version.

    Returns:
        The local path of the notebook, e.g. "<local_dir>/pfund_official/default_backtest.py".

    This is the seam: today the source is read from the bundled wheel; in pfund_hub
    the body becomes a remote fetch without changing callers.
    """
    filename = name if name.endswith(".py") else f"{name}.py"
    if "/" not in filename.strip("/"):
        raise ValueError(
            f"notebook name must be '<group>/<notebook_name>', got {name!r} "
            + "(e.g. 'pfund_official/default_backtest' or 'pfund_official/default_backtest.py')"
        )

    source = files(_NOTEBOOKS_ROOT) / filename
    if not source.is_file():
        raise FileNotFoundError(f"no notebook named {name!r}")

    dest = local_dir / filename
    if not dest.exists() or force:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(source.read_bytes())
    return str(dest)


def download_notebooks(
    names: list[str],
    *,
    local_dir: str = "./pfund_hub/notebooks",
    force: bool = False,
    num_workers: int = 8,
) -> list[str]:
    """Download official notebooks into a local, user-owned dir.

    Args:
        names: Notebooks to download, each as "<group>/<notebook>" (the ".py"
            suffix is optional), e.g. ["pfund_official/default_backtest"]. The
            group is required; a bare "default_backtest" raises ValueError.
        local_dir: Directory to copy the notebooks into. Each notebook keeps its
            "<group>/<notebook>.py" sub-path under here, so groups never collide.
        force: If False (default), an already-downloaded notebook is left
            untouched so your local edits survive. If True, overwrite it with the
            pristine official version, discarding local changes.
        num_workers: Max notebooks fetched concurrently.

    Returns:
        The local path of each downloaded notebook, in the order of `names`.

    Running the notebooks is the user's job, e.g.:

        marimo edit ./pfund_hub/notebooks/pfund_official/default_backtest.py

    Raises:
        ImportError: If marimo is not installed (needed to run the notebooks).
    """
    if importlib.util.find_spec("marimo") is None:
        raise ImportError(
            "marimo is required to run the notebooks but is not installed. "
            + "Install it with: pip install marimo"
        )

    dest = Path(local_dir)

    def _fetch(name: str) -> str:
        return _download(name, local_dir=dest, force=force)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        paths = list(executor.map(_fetch, names))

    print(
        f"Downloaded {len(paths)} notebook(s) to {dest}/. Run one with:\n"
        + f"    marimo edit {paths[0]}"
        if paths
        else f"No notebooks downloaded to {dest}/."
    )
    return paths
