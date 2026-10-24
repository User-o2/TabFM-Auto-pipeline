# seismic-bumps — final pipeline

Metric: `1-auc`, lower is better.

## What it does

Frozen TabFM does all the predicting; the pipeline only changes the shape of the
table it attends over. Three things happen, in this order.

**Columns are removed.** Of the 18 original features only 8 survive: `seismic`,
`shift`, `genergy`, `gpuls`, `gdenergy`, `gdpuls`, `nbumps`, plus the derived one
below. `ghazard`, `seismoacoustic`, `energy`, `maxenergy`, `nbumps2`–`nbumps5` are
dropped. This was found incrementally (dropping `seismoacoustic` was worth −0.0036,
dropping `energy` −0.0042) and then confirmed by a leave-one-out sweep over the
survivors: putting any one of the 8 back out costs between +0.0008 (`shift`) and
+0.0228 (`nbumps`), so the set is minimal, not merely small.

**One column is added.** `g_per_pulse = log1p(genergy / max(gpuls, 1))` — the energy
carried per seismic pulse, which the raw pair only encodes jointly (−0.0037). Other
ratios that looked equally natural were tried and hurt: energy-per-bump +0.0075,
bumps-per-pulse +0.0131.

**Every numeric column is gaussianised.** Each column is replaced by its percentile
against a reference distribution pooled over train **and** test rows (features only —
no labels are touched), then pushed through the inverse normal CDF. The seismic
features are heavy-tailed and orders-of-magnitude apart; this puts them all on one
comparable, roughly-normal axis, which is what TabFM's in-context attention reads
best. Pooling the reference over train+test also removes the train/test scale shift
for free.

TabFM's own knobs are almost untouched — only `norm_methods = ['power',
'quantile_rtdl']`. That is deliberate: the run tested feature crosses (twice), SVD
features, class shift, logit averaging, outlier thresholds, per-member row bagging,
ordinal category codes, minority-row repetition in the context (+0.024) and a rank
blend with RF/ET/LR, and every one of them was neutral or worse. On this task the
table's shape is the only lever that moves.

`folds()` uses `RepeatedStratifiedKFold(3 splits × 10 repeats)`. The positive class is
rare enough that a single 3-fold split has a fold-to-fold spread (std ≈ 0.022) far
larger than the effects being chased; 30 paired folds bring the standard error on a
paired comparison down to ~0.0012, which is what made ~0.003-sized changes
distinguishable at all. Rows are exchangeable, so no group or time structure is
imposed.

Validation score of this pipeline (eval 79, 30 folds, 8 estimators): `1-auc = 0.20298`,
against `0.21939` for the identity pipeline on the same folds.

## What build.sh builds

Nothing. `final/build.sh` prints a line and exits 0.

The pipeline reads no files: all four hooks operate purely on the DataFrames the judge
passes in, and no cache, embedding or fitted artefact is written to disk anywhere in
the run. Build time and disk cost are both zero.

## Things a reader could get wrong

- **`final/build/` is not part of the build.** `batch.py` was the edit-and-rerun helper
  that rewrote the `_ADD` / `TABFM_KWARGS` / `_DROP` lines of `pipeline.py` between
  evaluations; running it would overwrite the scored pipeline. `cmp.py` is read-only
  reporting over `log.txt`. Neither produces an input to `pipeline.py`, so `build.sh`
  documents them and does not call them.
- **`_ADD` / `_DROP` are a switchboard, not dead code.** `_shape()` and `engineer()`
  contain every feature variant tried during the search, each behind an `if '<name>' in
  _ADD`. Only `_ADD = ['log_gpp', 'tranks']` is live: `log_gpp` builds the logged
  energy-per-pulse column, `tranks` runs the train+test-pooled gaussian rank transform.
  Everything else in those functions is inert with this `_ADD` and is kept as the record
  of what was rejected.
- **`log_gpp` and `g_per_pulse` write the same column name.** They are alternatives —
  the plain ratio and its `log1p` — not two columns. Only one is ever in `_ADD`.
- **The rank transform uses test features, and that is intended.** `_rankify(...,
  pooled=True)` concatenates the train and test values of a column to build the
  reference quantiles. It sees no test labels, only the feature matrix the judge already
  hands `engineer()`.
- **`preprocess` and `postprocess` are deliberately identity.** Probability calibration
  is irrelevant to AUC, which is rank-only, so there is nothing for `postprocess` to do.