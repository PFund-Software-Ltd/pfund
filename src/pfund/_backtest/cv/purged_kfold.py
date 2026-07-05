"""
Purged K-Fold cross-validation.  (NOT YET IMPLEMENTED)

Purged K-Fold (López de Prado, "Advances in Financial Machine Learning", ch. 7)
adapts K-Fold to time series with *overlapping labels*: it "purges" training
samples whose label window overlaps the test fold, then applies an "embargo" gap
after each test fold, to prevent look-ahead leakage.

DO NOT re-derive the purging/embargo logic from the book — reuse a mature,
sklearn-compatible implementation instead:
  - skfolio.model_selection.PurgedWalkForward (actively maintained, https://skfolio.org)
  - historical reference: mlfinlab.cross_validation.PurgedKFold
      (now behind Hudson & Thames' paid tier)

Whatever we adopt exposes the sklearn splitter API
(`.split(X, y, groups) -> Iterator[(train_idx, test_idx)]`), but it does NOT drop
straight into `pfund._backtest.cv.base.fold_cv_region`: that helper reports each
train set as `(train_idx[0], train_idx[-1])`, a contiguous outer span, whereas
purging removes samples from the *middle* of the train set. That span would hide
the purge gaps, so the per-fold date-span model needs rethinking here.

PREREQUISITE: purging needs per-sample label end-times (`t1`) and an embargo
fraction — metadata the range-based `DatasetSplitter` does not have. Wiring that
through from the model/labeling layer is required before this is implementable.
"""

from __future__ import annotations
