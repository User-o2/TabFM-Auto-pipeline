# healthcare_insurance_expenses — final pipeline

## What it does

`charges` in this dataset is not a noisy function of the six covariates: it is a *closed form* in
them, plus a one-sided contamination. A median (LAD) regression on
`[age², age, children, bmi, male, region dummies, smoker, smoker·bmi, smoker·1(bmi>30)]` reproduces
the target to floating-point precision on ~91% of the training rows. The remaining ~9% sit 2k–24k
currency units *above* that surface, and nothing in the features predicts which rows those are
(OOF AUC ≈ 0.47 for the spike indicator; the rate is flat across every slice). Under RMSE the
optimal prediction is therefore `base(x) + E[spike]`, and the whole design is about handing TabFM a
context in which saying that is easy.

Concretely, `preprocess` fits the LAD hyperplane (`sklearn` `QuantileRegressor`, `quantile=0.5`,
`alpha=0`, `highs` solver, on standardized design columns; a trimmed-least-squares loop is the
fallback if the solver fails), computes `E[spike]` as the mean residual, and shrinks it 5% toward
zero to account for the estimation variance of a mean taken over a 9%/91% mixture. In the chosen
`MODE='clean'`, the spike rows (residual > 500) are then **dropped from the context**, so TabFM sees
a noiseless map, and the fitted `base` value is passed in as the single feature column
(`FEATS='base'`). `postprocess` adds the shrunken `E[spike]` back onto TabFM's output — that offset
is the one piece of information the dropped rows carried. The base column and the offset travel
from `preprocess` to `postprocess` through `.base_cache.npz`, because the two hook stages run in
separate interpreters.

TabFM remains the predictor throughout: it maps `base → charges` over the clean context rows.
Around it, `TABFM_KWARGS={'norm_methods': ['none']}` (the target is already on a natural scale), and
`sample()` returns 12 independent 90% subsamples of the context, whose predictions the judge
averages. `folds()` overrides the judge's single 3-fold with two shuffled 10-folds — the surviving
candidates were within a couple of RMSE units of each other, so the split had to be measured twice
to rank them, and 803 context rows per fold is much closer to the 892 the final model actually fits
on.

Metric is RMSE, lower is better; `data/meta.json` lists no medal thresholds for this task.

## What build.sh builds

Nothing. `final/build.sh` is a no-op that explains why and exits 0. The pipeline has no
prerequisites: its only file, `.base_cache.npz`, is written by `preprocess` on every run, and all
features are derived in-process from the raw `data/train.csv` / `data/test.csv` columns the judge
passes to the hooks. Runtime ~0 s, disk ~0 B. There is no `final/build/` directory because no
helper script was written during the run — every experiment was an edit to `pipeline.py` followed by
`python judge.py val`.

## Things a reader could get wrong

- **`.base_cache.npz` is not an input.** `final/run.py`'s static prerequisite scan sees the name as a
  string constant, but it is written before it is read within the same run. Do not try to supply it,
  and do not be surprised that it appears in the workspace after a run.
- **The spike rows are dropped, not down-weighted.** `MODE='clean'` removes ~9% of the training rows
  from TabFM's context entirely. Their contribution comes back only as the scalar `off` added in
  `postprocess`. `MODE='direct'`, `'resid'` and `'winsor'` are the alternatives that were tried and
  left in the file; changing `MODE` changes what `postprocess` must do, and the two are coupled.
- **`ALPHA=1.0` means the analytic base does not appear directly in the output** — it reaches the
  prediction only through the feature TabFM conditions on. Lowering `ALPHA` blends the raw LAD fit
  back in.
- **`OFF_SHRINK=0.05` is a fixed 5% shrink**, not the `'auto'` variance-based rule also implemented
  on the same line; `'auto'` would compute `off²/(off²+se²)`.
- `SETS` in `engineer` lists a dozen feature subsets explored during the run; only `FEATS='base'` is
  live. The engineered columns are all terms of the closed form — dropping them and giving TabFM the
  six raw columns alone (`'rawonly'`) costs about 60 RMSE.