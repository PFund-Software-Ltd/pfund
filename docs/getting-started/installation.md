# Installation

![GitHub stars](https://img.shields.io/github/stars/PFund-Software-Ltd/pfund?style=social)
![PyPI downloads](https://img.shields.io/pypi/dm/pfund)
[![PyPI](https://img.shields.io/pypi/v/pfund.svg)](https://pypi.org/project/pfund)
![PyPI - Support Python Versions](https://img.shields.io/pypi/pyversions/pfund)
[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)


## Using [Poetry](https://python-poetry.org) (Recommended)
```bash
# [RECOMMENDED]: Trading + Backtesting + Machine Learning + Feature Engineering (e.g. feast, tsfresh, ta) + Analytics
poetry add "pfund[all]"

# [Trading + Backtesting + Machine Learning + Feature Engineering]:
poetry add "pfund[data,ml,fe]"

# [Trading + Backtesting + Machine Learning]:
poetry add "pfund[data,ml]"

# [Trading + Backtesting]:
poetry add "pfund[data]"

# [Trading only]:
poetry add pfund

# update to the latest version:
poetry update pfund
```


## Using Pip
```bash
# same as above, you can choose to install "pfund[all]", "pfund[data,ml,fe]", "pfund[data,ml]", "pfund[data]" or "pfund"
pip install "pfund[all]"

# install the latest version:
pip install -U pfund
```


## Checking your installation
```bash
$ pfund --version
```