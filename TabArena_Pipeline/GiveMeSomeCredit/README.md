# GiveMeSomeCredit — final pipeline

## What it does

A frozen TabFM predicts credit default by in-context learning over the raw training rows; the
pipeline around it is deliberately thin. Two of the ten original columns are repaired before the
model sees them. `DebtRatio` is not one variable: for the ~80% of rows that report a
`MonthlyIncome` it is a debt-to-income ratio, and for the rest it is a dollar amount (median 1149
against 0.30), so the dollar rows are pushed through their own ECDF onto the ratio branch's
quantiles and the column becomes a single ordering again. Separately, TabFM's numeric front end is
a plain mean-imputer, so a missing `MonthlyIncome` would otherwise vanish entirely; an explicit
`income_missing` indicator restores it. No columns are added or dropped beyond that — every other
engineered feature tried (delinquency totals, crosses, SVD components, log/ratio bundles) lost.
On the model side, the eight ensemble members cycle only the two rank-based normalisations
(`quantile`, `quantile_rtdl`) instead of the default `none`/`power`, member outputs are averaged as
probabilities rather than logits, and `max_num_rows` is set to 80% of the training rows so the
members bag distinct contexts rather than all sharing one. CV metric `1-auc` (lower better): the
identity pipeline scores 0.13403, this one 0.13305.

## What build.sh builds

Nothing. `final/build.sh` is a no-op that exits 0. The pipeline reads no cache, index or
precomputed feature file — everything it needs is derived from the training DataFrame on each
call — and the run produced no helper scripts, so `final/build/` is absent by design. Zero extra
disk, zero build time.

    bash final/build.sh && python final/run.py

The single run is one TabFM pass with 8 estimators over 100k training rows; it needs a GPU and
takes on the order of minutes, not hours.

## Things a reader could get wrong

- **`max_num_rows` is computed, not fixed.** `preprocess` sets it from `len(X_train)`, because a CV
  fold holds ~66.7k rows while the final fit sees 100k, and the tuned invariant is the *fraction*
  (0.8), not a constant row count. Change `CONTEXT_FRACTION`, not the resolved number. This also
  means the pipeline mutates the module-level `TABFM_KWARGS` dict at call time — the judge reads it
  after the pre-hooks return, which is why that works.
- **The ECDF mapping is fit on train and applied to both splits.** `engineer` derives `src`/`dst`
  from the training rows only and reuses them for the test rows, so there is no leakage and test
  rows outside the training range clamp at the endpoints (`np.interp` behaviour, intended).
- **The 96/98 codes in the delinquency counters are left alone.** They look like not-available
  sentinels and NaN-ing them is tempting, but that group defaults at ~51%, and sending it to the
  column mean cost 0.00008. Same for `RevolvingUtilization >= 13`.
- **Resolution floor.** The judge is deterministic, so paired diffs are real, but any change to the
  column count or normalisation order reshuffles which config each member draws; differences below
  ~0.00005 are meaningless. Fold spread is ~0.0033.
- `run.py` copies `final/pipeline.py` over the workspace `pipeline.py` (refusing if the two differ)
  because the judge loads the hooks by that path.