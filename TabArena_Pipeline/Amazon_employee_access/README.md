# Amazon employee access — final pipeline

## What it does

The table is nine high-cardinality categorical id columns and a binary target; the metric is
`1 - ROC-AUC` (lower is better). TabFM pools labels by itself when it attends over the context
rows, so nothing label-derived is handed to it — out-of-fold target rates made it clearly worse.
What it cannot recover on its own is label-free co-occurrence structure, so `engineer` gives it
the raw ids plus three blocks computed over train and test together: the log count of every
single column value and every pair of column values, and, for every ordered pair of columns, how
many distinct values of the second each value of the first touches. Near-duplicate and constant
columns are dropped (the role hierarchy is nested, so a mechanical feature grid produces exact
copies). Wider blocks — triple counts, pair PMIs, a co-occurrence SVD, group means — were each
tried and each cost accuracy, so they are left out rather than pruned later.

Two side models, fitted on the fold's training rows only, cover the part of the hypothesis space
TabFM is weakest in. A sparse one-hot logistic regression over single/pair/triple key groups
(values seen once share a per-group "rare" slot), averaged over five regularisation strengths,
supplies an explicit weight per category value. A gradient-boosting average over five
(iterations, learning rate, leaves) settings fits axis-aligned splits on the same dense count
block, stacked on the linear model's out-of-fold logit — its test-side column comes from the same
inner folds as the training side, so the feature means the same thing on both. `engineer` writes
their test-side predictions to `.aux_pred.npz`; `postprocess` reads that back (the hooks run in
separate processes) and blends all three in standardised-logit space at .50 TabFM / .24 linear /
.26 boosting. TabFM stays the largest single component, and the blend's ranking still correlates
~0.95 with TabFM's own. Validation: 0.1260 for the blend vs 0.1318 for TabFM on these features
alone, over the 3 CV folds.

## build.sh

Nothing. `final/build.sh` prints a note and exits 0. The pipeline reads no precomputed cache —
every feature and both side models are derived inside `engineer` from the frames the judge passes
in. No hand-built caches were made during the run, so `final/build/` is absent. Zero time, zero
disk.

## Things a reader could get wrong

- `.aux_pred.npz` at the workspace root is **not** a prerequisite. `engineer` writes it and
  `postprocess` reads it back within the same run, purely to cross the process boundary between
  the hooks. Do not pre-build it; a stale copy is rejected by the row-count / row-id guard in
  `postprocess`, which then returns TabFM's prediction unblended (silently better than wrong, but
  it is not the scored configuration).
- Because of that guard, and the single-class early return in `_aux`, `python judge.py check`
  and any small-slice run legitimately produce a TabFM-only prediction. That is expected, not a
  broken hook.
- Counts and degrees are computed over train **and** test rows jointly. This is transductive but
  label-free — no target information crosses the split.
- The runtime cost is mostly the side models: five liblinear fits plus three stacking fits on a
  wide sparse matrix, and five `HistGradientBoostingClassifier` fits, all inside `engineer`,
  before TabFM runs at all. On the full training set expect this to dominate the non-GPU time.
- `_frame` forces an object-dtype column Index on purpose: the judge host unpickles hook output
  without pyarrow, and a pandas 3 arrow-backed string Index cannot be loaded there. Do not
  "simplify" it away.