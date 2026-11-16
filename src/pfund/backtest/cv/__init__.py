"""Dataset splitting for backtesting: holdout and cross-validation.

Users pass ``dataset_splits`` (``DatasetSplits``) to the engine, a shorthand that
needs no knowledge of holdout or cross-validation, e.g. ``721`` or
``{"train": 0.8, "test": 0.2}``, or ``None`` for no split. ``to_recipe``
converts a ``DatasetSplits`` into a recipe, and from there two layers follow,
one column per splitting method:

    +----------+------------------------------+------------------------------------+
    |          | Holdout                      | Cross-validation                   |
    +----------+------------------------------+------------------------------------+
    | input    | DatasetSplits (int / dict / recipe)                               |
    | recipe   | Holdout                      | CrossValidator (Protocol) /        |
    |          |                              | CrossValidation (built-in)         |
    | resolve  | resolve_holdout(X, holdout)  | resolve_folds(X, cv)               |
    | result   | HoldoutSplit                 | tuple[CVFold, ...]                 |
    +----------+------------------------------+------------------------------------+

- input: an int or dict always becomes a ``Holdout``; a recipe passes through.
- recipe: HOW to split, not bound to any data. It exists before the feature
  dataset ``X`` does. With no split (``None``), the whole of ``X`` is the
  training set.
- resolve: applies a recipe to a materialized ``X``. Splitting happens over
  the unique ``date`` values, so all rows at one timestamp land on the same side.
- result: positional row selectors bound to ``X``. ``.split(data)`` applies
  them to any data row-aligned with ``X`` (e.g. labels ``y``).

``HoldoutSplit`` has train/val/test and covers every row exactly once.
Each ``CVFold`` has train/val only; folds may leave rows out (e.g. gaps,
purging), and cross-validation creates no final test set.

``CrossValidator`` is the sklearn splitter interface, so any sklearn-compatible
splitter (``KFold``, ``TimeSeriesSplit``, future purged K-Fold, ...) is accepted.
That is why folds are resolved by a free function rather than a method: pfund
does not own those splitter classes. Holdout follows the same shape for symmetry.
"""

from pfund.backtest.cv.cross_validation import (
    CrossValidation,
    CrossValidator,
    resolve_folds,
)
from pfund.backtest.cv.cv_fold import CVFold
from pfund.backtest.cv.dataset_splits import DatasetSplits, DatasetSplitsDict, to_recipe
from pfund.backtest.cv.holdout import Holdout, resolve_holdout
from pfund.backtest.cv.holdout_split import HoldoutSplit

__all__ = [
    "CVFold",
    "CrossValidation",
    "CrossValidator",
    "DatasetSplits",
    "DatasetSplitsDict",
    "Holdout",
    "HoldoutSplit",
    "resolve_folds",
    "resolve_holdout",
    "to_recipe",
]
