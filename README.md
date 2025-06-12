# PFund: A Complete Algo-Trading Framework powered by Machine Learning and Data Engineering, TradFi, CeFi and DeFi ready.

[![Twitter Follow](https://img.shields.io/twitter/follow/pfund_ai?style=social)](https://x.com/pfund_ai)
![GitHub stars](https://img.shields.io/github/stars/PFund-Software-Ltd/pfund?style=social)
![PyPI downloads](https://img.shields.io/pypi/dm/pfund)
[![PyPI](https://img.shields.io/pypi/v/pfund.svg)](https://pypi.org/project/pfund)
![PyPI - Support Python Versions](https://img.shields.io/pypi/pyversions/pfund)
![Discussions](https://img.shields.io/badge/Discussions-Let's%20Chat-green)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/PFund-Software-Ltd/pfund)
<!-- [![marimo](https://marimo.io/shield.svg)](https://marimo.io) -->
<!-- [![Jupyter Book Badge](https://raw.githubusercontent.com/PFund-Software-Ltd/pfund/main/docs/images/jupyterbook.svg
)](https://jupyterbook.org) -->
<!-- [![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/) -->

[TradFi]: https://www.techopedia.com/definition/traditional-finance-tradfi
[CeFi]: https://www.techopedia.com/definition/centralized-finance-cefi
[DeFi]: https://www.coinbase.com/learn/crypto-basics/what-is-defi
[pytrade.org]: https://pytrade.org
[dYdX]: https://dydx.exchange
[polars]: https://pola.rs/
[PFund.ai]: https://pfund.ai
[PFeed]: https://github.com/PFund-Software-Ltd/pfeed
[Bybit]: https://bybit.com/
[PyTorch]: https://pytorch.org/
[Poetry]: https://python-poetry.org
[Futu]: https://www.futunn.com
[FirstRate Data]: https://firstratedata.com
[Mantine UI]: https://ui.mantine.dev/

> **This library is NOT ready for use, please wait for 0.1.0 release.**

## Problem
Machine learning (**AI**) and data engineering (**Big Data**) fields are advancing every year, but everyday traders are **not able to enjoy the benefits** of these improvements, leading to a **widening gap** between retail traders and professional traders.

## Solution
A modern algo-trading framework is needed to **bridge the gap** between algo-trading, machine learning and data engineering, empowering retail traders with state-of-the-art machine learning models and data engineering tools so that traders only need to focus on strategy research and the framework takes care of the rest.

---
PFund (/piː fʌnd/), which stands for "**Personal Fund**", is an **algo-trading framework** designed for using **machine learning** models natively to trade across [TradFi] (Traditional Finance, e.g. **Interactive Brokers**), [CeFi] (Centralized Finance, e.g. Binance) and [DeFi] (Decentralized Finance, e.g. [dYdX]), or in simple terms, **Stocks** and **Cryptos**.

## Core Features
- [x] Supports vectorized and event-driven backtesting with different resolutions of data, e.g. tick data, second data and minute data etc.
- [x] Allows choosing your preferred data tool, e.g. pandas, polars, pyspark etc.
- [x] Supports machine learning models, features, technical analysis indicators
- [x] Trains machine learning models using your favorite frameworks, i.e. PFund is **ML-framework agnostic**
- [x] Offers **LEGO-style** strategy and model building, allowing strategies to add other strategies, models to add other models
- [x] Streamlines the algo-trading flow, from vectorized backtesting for strategy prototyping and event-driven backtesting for strategy development, to live trading for strategy deployment
- [x] Enables parallel data processing, e.g. Interactive Brokers and Binance each have their own process for receiving data feeds
- [x] Switches from backtesting to live trading by just changing **ONE line of code!!**
- [ ] Features a modern frontend using [Mantine UI] and TradingView's Charts library
- [ ] Supports manual/semi-manual trading via a trading app

> As PFund is for trading only, for all the data workloads, there is a separate library to handle that:\
[PFeed] - Data pipeline for algo-trading, helping traders in getting real-time and historical data, and storing them in a local data lake for quantitative research.

---

<details>
<summary>Table of Contents</summary>

- [Installation](#installation)
- [Quick Start](#quick-start)
  - [Backtesting](#backtesting)
  - [Live Trading](#live-trading)
  - [Parameter Training / Hyperparameter Tuning](#parameter-training--hyperparameter-tuning)
  - [Building LEGO-Style Strategy and Model](#building-lego-style-strategy-and-model)
- [PFund Hub](#pfund-hub)
- [Supported Trading Venues](#supported-trading-venues)
- [Related Projects](#related-projects)
- [Disclaimer](#disclaimer)

</details>


## Installation

### Using [Poetry] (Recommended)
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

### Using Pip
```bash
# same as above, you can choose to install "pfund[all]", "pfund[data,ml,fe]", "pfund[data,ml]", "pfund[data]" or "pfund"
pip install "pfund[all]"

# install the latest version:
pip install -U pfund
```

### Checking your installation
```bash
$ pfund --version
```


## Quick Start
### Backtesting
```python
import pfund as pf

# NOTE: for more exciting strategies, please visit pfund.ai
class YourStrategy(pf.Strategy):
    # triggered by bar/kline data (e.g. 1-minute data)
    def on_bar(self):
        # write your trading logic here
        pass


engine = pf.BacktestEngine(mode='vectorized')
strategy = engine.add_strategy(YourStrategy(), name='your_strategy')
strategy.add_data(
  'IB', 'AAPL', 'USD', 'STK', resolutions=['1d'],
  backtest={
    # NOTE: since IB does not provide any historical data for backtesting purpose, use data from 'YAHOO_FINANCE'
    'data_source': 'YAHOO_FINANCE',
    'start_date': '2024-01-01',
    'end_date': '2024-02-01',
  }
)
engine.run()
```


### Live Trading
> Just change one line of code, from '**BacktestEngine**' to '**TradeEngine**'. BOOM! you can now start live trading.
```python
import pfund as pf

engine = pf.TradeEngine(env='LIVE')
strategy = engine.add_strategy(YourStrategy(), name='your_strategy')
strategy.add_data(
  'IB', 'AAPL', 'USD', 'STK', resolutions=['1d'],
  # for convenience, you can keep the kwarg `backtest`, `TradeEngine` will ignore it
  backtest={
    # NOTE: since IB does not provide any historical data for backtesting purpose, use data from 'YAHOO_FINANCE'
    'data_source': 'YAHOO_FINANCE',
    'start_date': '2024-01-01',
    'end_date': '2024-02-01',
  }
)
engine.run()
```

### Parameter Training / Hyperparameter Tuning
> The correct term should be "Hyperparameter Tuning", but since not all traders are familiar with machine learning, the framework uses a more well-known term "training".

```python
import pfund as pf

engine = pf.TrainEngine()
strategy = engine.add_strategy(...)
strategy.add_data(...)
strategy.add_indicator(...)
engine.run()
```

### Building LEGO-Style Strategy and Model
```python
import pfund as pf

engine = pf.TradeEngine(env='LIVE')
strategy = engine.add_strategy(...)
strategy.add_data(...)
model = strategy.add_model(...)

model.add_data(...)  # using different data than strategy's
sub_model = model.add_model(...)  # YES, model can add another model to its use
# You can keep going: 
# sub_sub_model = sub_model.add_model(...)

engine.run()
```


## PFund Hub
Imagine a space where algo-traders can share their trading strategies and machine learning models with one another.
Strategy and model development could be so much faster since you can build on top of an existing working model.


---

## Supported Trading Venues
| Trading Venue             | Vectorized Backtesting | Event-Driven Backtesting | Paper Trading | Live Trading |
| ------------------------- | ---------------------- | ------------------------ | ------------- | ------------ |
| Bybit                     | 🟢                     | 🟡                       | 🟡            | 🟡           |
| *Interactive Brokers (IB) | 🟡                     | 🟡                       | 🟡            | 🟡           |
| Binance                   | 🔴                     | 🔴                       | 🔴            | 🔴           |
| OKX                       | 🔴                     | 🔴                       | 🔴            | 🔴           |
| *Alpaca                   | 🔴                     | 🔴                       | 🔴            | 🔴           |
| *[Futu]                   | 🔴                     | 🔴                       | 🔴            | 🔴           |
| dYdX                      | 🔴                     | 🔴                       | 🔴            | 🔴           |

🟢 = finished \
🟡 = in progress \
🔴 = todo \
\* = use a **_separate data source_** (e.g. [FirstRate Data]) for backtesting


## Related Projects
- [PFeed] — Data engine for algo-trading, helping traders in getting real-time and historical data, and storing them in a local data lake for quantitative research.
- [PyTrade.org] - A curated list of Python libraries and resources for algorithmic trading.


## Disclaimer
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

This algo-trading framework is intended for educational and research purposes only. It should not be used for real trading without understanding the risks involved. Trading in financial markets involves significant risk, and there is always the potential for loss. Your trading results may vary. No representation is being made that any account will or is likely to achieve profits or losses similar to those discussed on this platform.

The developers of this framework are not responsible for any financial losses incurred from using this software. Users should conduct their due diligence and consult with a professional financial advisor before engaging in real trading activities.