# customer_satisfaction_in_airline — final pipeline

## What it does

The prediction is made entirely by the frozen TabFM in-context learner over the raw training
rows; the pipeline around it is deliberately minimal, because on this task every attempt to
add structure lost to leaving the columns alone. The only feature change is a single
imputation: `ArrivalDelayinMinutes` is missing on a small number of rows, and it is filled
from `DepartureDelayinMinutes` on the same row, which predicts it far better than any column
statistic. Nothing else is transformed, no column is dropped, no target post-processing is
applied.

The gains instead come from how TabFM is asked to look at the data. The full training set is
used as context (`max_num_rows = 131072` > 86,586 training rows, so no subsampling happens),
and `sample()` returns two draws over the *same* rows — the natural order and its exact
reverse. That is not a bagging trick: it is a cheap way to average out TabFM's residual
sensitivity to context ordering while keeping every row in view. Row bagging on top of it was
tried and was 4e-5 worse — the two axes are not complementary, and context size wins. The
ensemble knobs mix four input normalisers (`none`, `power`, `quantile_rtdl`, `robust`) rather
than the default set, enable `permute_categorical`, and use `sqrt`-many SVD features; each was
ablated individually against 3-fold CV, and the normaliser mix and the SVD setting are the two
that carry most of the improvement. `outlier_threshold` was the last knob resolved: it reaches
only the two delay columns (nothing else has a z that large) and the clip is soft, so it just
sets where their tail starts compressing. Bracketing it at 8 / 12 / 16 / 30 gave a
single-peaked curve, and 10.0 is what came out of closing that bracket.

Validation: 3-fold CV, `1-auc` (lower is better), best ~0.00390. Held-out test: **0.003814**,
against a TabArena ensemble baseline of 0.003800 and a plain baseline of 0.003939.

## What build.sh builds

Nothing. The pipeline reads no caches or derived files — only the competition CSVs, which the
judge loads. `final/build.sh` is a comment and `exit 0`: it takes no time and needs no disk.

```
bash final/build.sh && python final/run.py
```

## Things a reader could get wrong

- **The reversed index in `sample()` is not a different subset.** Both draws contain all
  86,586 training rows. If you "simplify" it to a single draw you lose ~1e-5 of metric; if you
  read it as subsampling and try to make the two halves disjoint you will do considerably
  worse.
- **`max_num_rows = 131072` is not a subsample size here.** It exceeds the training set, so it
  is effectively "use everything". It matters only if this pipeline is pointed at a larger
  table.
- **The arrival-delay fill mutates a column in place** rather than adding an indicator. A
  missingness flag did not help; the information is apparently already carried by the departure
  delay.
- **`run.py` copies `final/pipeline.py` over the workspace `pipeline.py`** and refuses to run
  if the two differ. They are identical as shipped (both are `history/pipeline_061.py`), so
  this only bites if you edit one of them.
- **The estimator count is the judge's, not the pipeline's.** Scoring used 8
  (`judge.estimators_final`); `run.py -n` overrides it. Nothing in `TABFM_KWARGS` sets it.
- The TabFM pass itself is the whole cost: ~19 minutes wall clock on the scoring run
  (~1050 s of it in predict), dominated by the test-set forward passes.