## Installation
```bash
poetry add pfund --with dev,doc
```

## Build Documentation using [jupyterbook](https://jupyterbook.org/)
```bash
# at the root directory, run:
jb build docs/ [--all]
```

## Update submodules
```bash
# run this to see if the version a submodule is using has been changed.
git submodule update
```