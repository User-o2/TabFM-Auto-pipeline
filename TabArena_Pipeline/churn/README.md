# churn (TabArena) — final pipeline

## What it does

The prediction is made entirely by frozen TabFM in-context over the training rows; the pipeline only
changes what those rows look like. Three things happen to the feature frame. First, the four
`total_*_charge` columns are dropped: each is exactly its minutes column times a fixed per-second
rate (day .170, eve .085, night .045, intl .270, residual under a cent of rounding), and exact
duplicate columns are the one thing TabFM measurably disliked here. Second, the two sums no single
column can express are added back — `total_charge` and `total_minutes`. Third, four *linear summaries
of the whole row* are appended, all fitted on the current fold's training rows only and applied to
both sides: the two leading principal components, then an LDA direction (lsqr, shrinkage 0.05) fitted
on the frame those components are already part of, then an L2 logistic direction (C=0.5) fitted on the
frame including the LDA column. The two supervised directions are rank-transformed against the
training-row distribution before being handed over, so their in-context values sit on the same scale
as their test values. The only TabFM knob changed from default is `n_svd_features='sqrt'`, which gives
roughly half the ensemble members random projections on top of the columns. `preprocess` and
`postprocess` are the identity, and `tabfm_corr` is 1.000 — nothing blends into or overrides TabFM's
output.

The design constraint behind the summary columns is variance, not accuracy: a column may only be added
if its value on the rows it was fitted on means the same thing as its value on unseen rows. That is
what separated these four from the two strong-but-overfit alternatives that were tried and rejected —
a stacked GBDT column (+0.008) and a logistic on all 231 pairwise interactions (+0.007).

Validation used `RepeatedStratifiedKFold(5 × 8, seed 1)` = 40 folds, because the fold-to-fold spread
(~0.017) is an order of magnitude larger than the effects being chased (0.001–0.003); every decision
above is a paired comparison across those 40 folds. Score on them: 0.07341 identity → 0.07109 base
features → 0.06486 this pipeline (1 − ROC-AUC, lower is better).

## What build.sh builds

Nothing. `final/build.sh` is a comment and `exit 0`. The pipeline reads no files and names no paths —
every derived column, including the fitted PCA/LDA/logistic directions, is computed inside the hooks
from the frames the judge passes in, per fold. No caches, no artifacts, no disk, no time.

    bash final/build.sh && python final/run.py

## Things a reader could get wrong

- **The summaries are refitted per call, not once.** `engineer()` fits PCA, LDA and the logistic on
  whatever `X_train` it is handed. During CV that is the fold's training rows; at test time it is the
  full training set. This is deliberate — it is what keeps the columns honest — but it means the
  feature frame is not reproducible from a saved artifact, and a run costs one PCA + one LDA + one
  logistic fit on ~3k × 21 (milliseconds) on top of the TabFM pass.
- **`_summaries` has a degenerate branch.** If the labels handed in are all one class, the two
  supervised columns are filled with zeros. That exists only so `python judge.py check`, which runs a
  150-row slice, does not crash; it never fires on real data.
- **`SQUASH = 'rank'`** despite the neighbouring comment saying "sigmoid". Both were measured and tied
  (0.06499 vs 0.06502); rank is what the constant selects and what ran.
- **`final/build/cmp.py`** is not part of the pipeline. It is the analysis helper used during the run
  to compute paired diffs between `[RESULT n]` lines in `log.txt`; it needs `log.txt`, not the data.
- The header docstring of `pipeline.py` is the full experiment record for the run — 73 evaluations,
  including everything that did not work. It is worth more than this README if you intend to keep
  searching.