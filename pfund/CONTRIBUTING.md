## Installation
```bash
git clone git@github.com:PFund-Software-Ltd/pfund.git
cd pfund
git submodule update --init --recursive
poetry install --all-extras
```

## Pull updates
```bash
# --recurse-submodules also updates each submodule to the commit specified by the main repository,
git pull --recurse-submodules  # = git pull + git submodule update --recursive
```

## Build Documentation using [jupyterbook](https://jupyterbook.org/)
```bash
# at the root directory, run:
jb build docs/ [--all]

# check if external links are broken:
jb build docs/ --builder linkcheck
```