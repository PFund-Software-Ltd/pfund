"""
Combinatorial Purged Cross-Validation (CPCV).  (NOT YET IMPLEMENTED)

CPCV (López de Prado, "Advances in Financial Machine Learning", ch. 12) splits the
data into N groups, uses every C(N, k) combination of k groups as the test set,
and applies purging + embargo to each. Unlike K-Fold it produces MULTIPLE backtest
*paths* (not folds of a single partition), yielding a distribution of performance
rather than a single point estimate.

DO NOT re-derive the combinatorics + purging — reuse a mature, sklearn-compatible
implementation instead:
  - skfolio.model_selection.CombinatorialPurgedCV (actively maintained, https://skfolio.org)
  - historical reference: mlfinlab.cross_validation.CombinatorialPurgedKFold
      (now behind Hudson & Thames' paid tier)

CPCV breaks the current one-return-shape assumption: it emits paths, so the output
type will differ from `CrossValidatorDatasetPeriods` (which is per-fold). Design the
path-assembly type here when implementing; `fold_cv_region` is per-fold and will
NOT map cleanly onto paths.

PREREQUISITE: same as purged K-Fold — needs per-sample label end-times (`t1`) and
an embargo fraction, which the range-based `DatasetSplitter` does not have.
"""

from __future__ import annotations
