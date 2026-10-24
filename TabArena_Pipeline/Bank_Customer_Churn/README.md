# Bank_Customer_Churn — final pipeline

Metric is `1 - ROC-AUC` (lower is better) over 6666 training rows and 10 raw columns.

## What it does

TabFM does all the predicting; the pipeline's whole contribution is deciding what table it gets to
attend over. The table is made *shorter and less numeric*, not richer. Three columns are dropped
(`estimated_salary`, which is uniform noise by construction; `credit_card`, 0.214 vs 0.200 churn;
`tenure`, flat at 0.177–0.226 across all 11 values with no trend). `products_number` and
`active_member` are recoded as string levels, because churn against `products_number` goes
28% / 7% / 83% / 100% and presenting that as an ordinal number asks the model to un-learn the
ordering. `balance` is a spike-at-zero mixture — 43% of rows are exactly 0.0, the rest roughly
normal around 120k — so it is split into a NaN-for-zero column (TabFM handles NaNs natively) plus an
explicit `bal_zero` flag; both halves earn their place, dropping either is worse. On the ensemble
side only two constructor knobs beat the defaults: `class_shift=False` (it adjusts members for the
20/80 class prior before averaging, so unlike the monotone knobs it can move a rank metric) and
`norm_methods=['none']` instead of `['none','power']`, the power-transformed half of the members
being the weaker one. `folds()` is 5×repeated stratified 3-fold: the judge's default single 3-fold
has std 0.009 on a 0.126 score and cannot rank changes worth 0.0005, while the repeats keep the 2:1
train/validation ratio the final is scored at. No other model appears anywhere — GBDT stacking and
rank-blending were tried and lost at every weight. Identity scored 0.12761 on these folds; this
scores 0.12576, better on 15/15 folds.

## What build.sh builds

Nothing. `pipeline.py` reads no files — its hooks are pure functions of the DataFrames the judge
passes in, and there is no `sample()` hook. `final/build.sh` is a documented no-op that exits 0 in
well under a second and needs no disk. Run:

```
bash final/build.sh && python final/run.py
```

## Things a reader could get wrong

- **`final/build/` is not called by `build.sh`, on purpose.** `xp.py` is an experiment driver that
  *rewrites module-level constants inside `pipeline.py`*; running it now would mutate the scored
  pipeline. `guard.py` re-ran the hooks and pickled the output to `$TMPDIR/guard.pkl` purely so the
  judge's venv could attempt to unpickle it; that file is scratch and the pipeline never reads it.
  Both are kept as the record of how the search was run.
- **The `_object()` helper is load-bearing, not style.** The hooks execute in a pandas 3.0 that has
  pyarrow, where an ordinarily-assigned string column becomes an `ArrowStringArray`; the judge
  unpickles the result in a pandas 3.0 *without* pyarrow and refuses to load one. `judge.py check`
  runs in-process, never crosses that boundary, and reports ok either way. Forcing plain object
  dtype is what keeps the handoff working.
- **The dropped columns are a measured choice, not a rule about flat features.** `credit_score`
  looks just as flat by decile as the dropped ones, but removing it costs 0.0013, so it stays.
- Additive feature engineering was tried extensively (interactions, ratios, age bins, zero-balance ×
  country) and every variant lost by 0.0001–0.0011. Oversampling the minority class in the context
  rows is catastrophic (0.147 at 2×, 0.289 at 3×). The docstring at the top of `pipeline.py` is the
  full negative-results list; consult it before re-trying anything.