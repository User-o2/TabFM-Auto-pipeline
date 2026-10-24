# miami_housing — final pipeline

## What it does

TabFM predicts `SALE_PRC` directly from the Miami housing table; the pipeline never
transforms the target and never post-processes the prediction. All of the work is in how
the *features* are presented to the in-context model, and in TabFM's own ensemble knobs.

Two things happen to the features. First, the thirteen columns that are non-negative
(`LND_SQFOOT`, `TOT_LVG_AREA`, `SPEC_FEAT_VAL`, the five distance columns, `age`,
`month_sold`, `structure_quality`, `avno60plus`) get `log1p`. `LATITUDE` and `LONGITUDE`
are deliberately left alone — a log of latitude in this region is effectively a constant,
and longitude is negative. Second, each logged column is shifted by a constant offset of
`1024 × std(column)`, the same offset applied to train and test.

That shift is the non-obvious part, and it is what most of the search converged on. It is
invisible to TabFM's rank/quantile-based normalisers — a monotone constant offset does not
move a quantile — but it is not invisible to the *feature crosses*, which are products of
raw columns. Pushing a column far from zero drives its pairwise products toward the
additive regime: `(a+s)(b+s)/s² ≈ 1 + (a+b)/s`, so a cross degenerates into a scaled sum
of its two parents plus a vanishing interaction term. Larger shifts kept helping
monotonically across the search (16 → 64 → 256 → 1024) with roughly constant gains per
doubling in `log s`, and 1024 is the last rung before float32 precision becomes the
limiting factor: at that scale a cross varies by about 2/s ≈ 2e-3 relative to its own mean.

TabFM's configuration mirrors this: `n_feature_crosses='sqrt'` so crossed members exist at
all, `outlier_threshold=20.0` so the shifted values are not clipped, and a four-element
normaliser cycle `['quantile', 'quantile_rtdl', 'quantile_rtdl', 'power']` whose *order*
is load-bearing — it puts the two rank-based methods on the plain (even-indexed) members,
where the shift is a no-op, and `quantile_rtdl`/`power` on the crossed (odd) ones, where it
is the whole point.

`folds()` returns three repeats of 3-fold (`RepeatedKFold`, seed 0): nine splits, each
training on 2/3 of the rows, which is what the final fit sees. Rows are exchangeable here,
so no group or time structure is imposed.

Result on the 9 TabArena folds: **RMSE 71802.2** mean, against the published TabFM
baselines of 74223.7 (plain) and 73559.4 (ensemble); CV during the search read 74765.0.
`tabfm_corr = 1.0` — TabFM is the sole predictor, as required.

## What build.sh builds

Nothing. The pipeline reads no file it does not itself write, so `final/build.sh` is a
comment and `exit 0`. It runs instantly and needs no disk. Running it is still the
documented entry point:

    cd <this workspace> && bash final/build.sh && python final/run.py

`python final/run.py` is the real cost: one frozen 8-estimator TabFM pass over the task.
On the H100 this run used, the scored final took ~75 s of wall clock across all 9 folds
(2.6 s preprocess, 4.1 s fit, 58.6 s predict, 2.4 s postprocess, over 82656 train / 41328
test rows in total). It needs a GPU with roughly 12 GB free; the table is small enough that
the context never approaches the judge's row cap.

## What a reader might otherwise get wrong

- **The data is a judge prerequisite, not a pipeline one.** This is a TabArena task:
  `judge.py` reads `${TABARENA_DIR:-$HOME/.cache/tabarena}/miami_housing/fold_<k>/{train,test}.csv`
  for k = 0..8 and asserts `fold_0/train.csv` exists. That cache is produced by the repo's
  TabArena downloader, which was never invoked from this workspace — so the exact command
  is *not* part of this run's record and `build.sh` does not guess at it. Populate the
  cache the same way the rest of the repo does before running `final/run.py`.
  `data/train.csv` and `data/test.csv` here are fold_0 materialised for the agent to read;
  they are not what the final is scored on.
- **The final is a mean over 9 folds, not one holdout.** `run.py`/`judge.py` refit the same
  pipeline independently on each fold. Per-fold RMSE ranged 64276 to 78218 — a spread of
  ~14k — so a single-fold number is not comparable to the headline 71802.
- **The 1024 shift is not a normalisation trick and does not commute with anything.** It is
  applied *after* the log and *before* TabFM, and it only means something in combination
  with `n_feature_crosses='sqrt'` and the normaliser order above. Reordering `norm_methods`,
  dropping the crosses, or lowering `outlier_threshold` changes what the shift does, not
  just how much.
- **1024 was the end of a trend, not a measured optimum with headroom above it.** The search
  never saw the curve turn; it stopped there because float32 precision in the crosses
  becomes the binding constraint, and this was the last rung worth trusting on a final that
  could not be re-measured.