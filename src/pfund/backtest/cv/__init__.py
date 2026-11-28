"""Dataset splitting for backtesting: holdout and cross-validation.

Users pass ``dataset_splits`` (``DatasetSplitsInput``) to the engine; ``None`` means
100% train, the whole dataset is the training set. There is always a split, so what a
component ran on is always known: with 100% train the train set is in use by default,
with any other split, call ``engine.use_split()`` before accessing a component's ``X``
or saving a model.

The grid is one bar per ``split_resolution`` (backtest engine settings) over the
engine's data range. A sealed ``test`` tail is carved first, then ``folds`` are
cut over everything before it (``trainval``)::

    grid:     |<-------------------- trainval -------------------->|<-- test -->|
    holdout:  | train                             | val            |  test      |
    cv fold 0:| train      | val |                                 |  test      |
    cv fold 1:| train            | val |                           |  test      |
    cv fold 2:| train                  | val |                     |  test      |

- recipe: HOW to split, bound to no data. ``CrossValidator`` is the sklearn
  splitter interface, so any sklearn-compatible splitter (``KFold``,
  ``TimeSeriesSplit``, future purged K-Fold, ...) is accepted.
- ``DatasetSplits``: every part is a set of grid positions, so a fold may leave gaps
  (e.g. purging) or have train on both sides of val (``KFold``).
- component: maps its own rows onto the grid by ``date``, so every component
  shares the same boundaries; ``split_resolution`` must not be finer than its
  resolution, or a bar would straddle a cut. Components never split on their own.
"""

from pfund.backtest.cv.cross_validation import CrossValidator, WalkForward
from pfund.backtest.cv.dataset_splits import (
    DatasetSplits,
    Fold,
)
from pfund.backtest.cv.holdout import Holdout

__all__ = [
    "CrossValidator",
    "DatasetSplits",
    "Fold",
    "Holdout",
    "WalkForward",
]
