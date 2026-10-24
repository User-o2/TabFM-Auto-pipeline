# in_vehicle_coupon_recommendation — final pipeline

Metric: `1-auc`, lower is better. Cross-validated score of this pipeline: **0.15261** (5-fold;
the untouched identity pipeline scores 0.17055 on the judge's 3 folds).

## What it does

The prediction is made entirely by frozen TabFM over the raw training rows — there is no second
model, no sampling of the context, and no post-processing (the metric is AUC, so only a
non-monotone transform could move it, and per-coupon recalibration was measured to hurt). All the
work is in how the survey is represented to TabFM and in how its ensemble is configured.

Feature side, in `engineer`: the ordered survey bands are recoded as numbers rather than left as
strings, because their order is invisible to TabFM while they are categorical — the five
visit-frequency columns onto a 0–4 rank scale, plus age, income, education and the coupon
expiration onto natural units. Blank bands are kept visible with a sentinel below `never` and a
per-row missing count, since TabFM mean-imputes numeric columns and would otherwise erase them.
The two nested distance flags collapse into one 0/1/2 level; `direction_opp` (the complement of
`direction_same`) and the ~99%-missing `car` column are dropped. Three columns tie the coupon to
the respondent's habits: how often they visit the venue type the coupon is for, their total
going-out frequency, and the ratio of the two. Finally six string-concatenated interaction
columns anchored on `coupon` (× destination, passenger, time, expiration, CoffeeHouse, income) —
TabFM's own `n_feature_crosses` only pairs numeric columns, so categorical × categorical
interactions are the ones it cannot construct for itself; the set was ranked by a logistic probe
over one-hot main effects and confirmed by a leave-one-out at the end.

Model side, `TABFM_KWARGS` turns on feature crosses and SVD features (both `sqrt`), categorical
permutation, and a five-method normalisation cycle. The cycle length is deliberately odd:
ensemble members alternate augmented/plain by index parity, so an even-length cycle would pin
each normalisation to one parity forever.

`folds()` overrides the judge's 3 folds with 5 stratified folds — the search reads paired
differences between pipelines, and 5 folds both halve the noise on that difference and put the
per-fold training size closer to the ~11.4k rows the final actually fits on.

The pipeline's own docstring carries the full negative record (respondent-level features, model
blending, context sampling, NNLS member weights, …) so those are not retried.

## build.sh

There is nothing to build. `pipeline.py` reads no files — every feature is computed row-wise from
the DataFrames the judge passes in. `final/build.sh` is a documented no-op that exits 0 in well
under a second and needs no disk. There is no `final/build/` directory because no helper script
was ever written during the run; all exploration was throw-away shell heredocs.

```
cd <workspace> && bash final/build.sh && python final/run.py
```

## Things a reader could get wrong

- **`sample` and `postprocess` are absent/identity on purpose.** That is a measured choice, not an
  omission: the judge's default context (one draw, all training rows) beat `max_num_rows=4000` by
  0.0025, and every re-weighting of the hard rows was worse.
- **`softmax_temperature` and `binary_calibration_method` are missing for a reason.** With
  `average_logits=True` the temperature is a monotone rescaling of the averaged logits and Platt
  calibration runs after averaging; both are exactly rank-preserving and so cannot move AUC.
- **The rows are not exchangeable at the respondent level** — 567 respondents answered ~22
  scenarios each, and every profile in the test split also occurs in training. That looks like free
  signal but is not: any per-respondent encoding cost 0.03–0.05 AUC. `folds()` therefore uses plain
  stratified splits, matching how the test split was made.
- **`run.py` copies `final/pipeline.py` over the workspace `pipeline.py`** and refuses to run if the
  two differ, so a reproduction cannot silently score a different file.