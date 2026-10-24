# polish_companies_bankruptcy — scored pipeline

Metric is `1 - AUC` (lower is better). Test score of this pipeline: **0.005178**, mean over the 9
TabArena folds (per-fold spread 0.0025–0.0075). The published TabFM baselines on the same folds are
0.005806 plain / 0.004802 ensemble. Search best under 5×2 repeated stratified CV was 0.00535.

## What it does

TabFM is the only predictor; everything here is about the table it gets to attend over. The 64
financial-ratio columns are heavy-tailed and span many orders of magnitude, so each one is passed
through a signed log, `sign(x)·log1p(|x|)`, which compresses the tails while keeping the sign — on
bankruptcy ratios (negative equity, negative working capital, negative profit) the sign carries real
information, so a plain log or a rank transform throws something away. The ratio set is also highly
redundant: a greedy pass drops any column whose absolute Pearson correlation with an already-kept
column exceeds 0.98, which shortens the context TabFM has to read without losing a direction of
variation. Missingness is treated as signal rather than noise: NaNs are left as NaNs for TabFM to
handle, and the pattern is made explicit with one binary indicator per column that has any missing
value plus a single `n_missing` count per row. Finally, `feat_shuffle_method` is set to `none`, so
TabFM's ensemble members all see the columns in the same, natural order instead of independently
permuted ones — on a table this small (3,940 train rows) that turned out to help rather than hurt.

No row sampling, no reweighting, no calibration: the whole training set is one draw, and the
predicted probabilities are returned untouched (AUC is rank-only, so a monotone postprocess could not
change the score). The `engineer()` body carries several other switches — winsorizing, MAD scaling,
cube root, rank→Gaussian, median/sentinel fill, correlation-cluster averaging, MI column ordering,
log-ratio pair features, GBDT out-of-fold log-odds as a column, permutation-importance top-k
selection — but the scored config (`CFG = {'tf': 'slog', 'miss_ind': 1, 'n_miss': 1, 'dedup': 0.98}`)
leaves all of them off. They are dead code in this configuration, kept because the file is the exact
artifact that was scored.

`folds()` is 5-fold stratified, repeated twice (`random_state=12345`), which is what the search
validated against; rows are exchangeable here, so nothing beyond stratifying on the rare positive
class is needed.

## What build.sh builds

Nothing. `final/build.sh` is a no-op that says so and exits 0 — the pipeline reads no cache, no
precomputed feature file, no external artifact. It runs in well under a second and needs no disk.
Go straight to `python final/run.py` (a few minutes, dominated by the TabFM forward passes; the
recorded final took ~61 s of pipeline+model time at 8 estimators).

## Things a reader could get wrong

- **`final/build/gen.py` is not a data step.** It is a code generator: `python gen.py '<json cfg>'`
  writes `pipeline.py` from a template, and `sweep.sh` in the workspace called it once per candidate
  config so several evaluations could be queued per turn. The chosen config is already inlined in
  `final/pipeline.py`. Do not run gen.py as part of a reproduction — it writes `pipeline.py` into the
  current directory and would clobber the scored record.
- **The scored snapshot is `history/pipeline_065.py`**, copied here as `final/pipeline.py`. Later
  snapshots exist in `history/` (up to 096) and are *not* what was tested.
- **`run.py` copies `final/pipeline.py` over the workspace `pipeline.py`** because the judge's
  `apply_hooks` loads it by name. It refuses to do so if the two differ, which is a safety check, not
  a failure of the build.
- The dedup threshold is applied *after* the signed-log transform and is fitted per fold on that
  fold's training rows, so the surviving column set is not fixed across folds. That is intentional;
  it keeps the selection inside the fold and out of the validation rows.