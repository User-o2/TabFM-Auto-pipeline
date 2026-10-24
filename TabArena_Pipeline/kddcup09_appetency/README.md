# kddcup09_appetency — final pipeline

## What it does

The task is ~33k rows × 212 columns (174 numeric, 38 string) with 1.8% positives, scored as
1 − AUC (lower is better). TabFM does the predicting; the pipeline's whole job is to decide
*which* columns it sees and how the high-cardinality string columns are represented.

For each of the 38 string columns it builds two numeric views from the training rows of the
current fold: a level **count** and a **target encoding** (positive rate per level, smoothed
toward the prior with `TE_SMOOTH = 60`). Both are made to look the same on the context side
and the query side, which was the single biggest win of the run: the target encoding is
cross-fitted over 5 random blocks (a context row's encoding is computed from the other four
blocks, so it never sees its own label), the count a context row reads likewise excludes its
own block and is rescaled by 5/4, and any level occurring fewer than `MIN_LEVEL = 10` times
collapses to the prior on both sides. Without that symmetry a rare level reads one way in
context and another way at query time and the score collapses (0.409 vs 0.152). String
columns with at most 30 levels are additionally passed through as raw categoricals.

The resulting ~273 columns are then ranked by in-fold univariate AUC — with NaN scored both
as the smallest and as the largest value, taking the better of the two, so a column whose
signal is purely its missingness pattern ranks on par with one whose signal is in its values —
and only the top `N_KEEP = 200` survive. Most of the raw columns are ~97% missing noise, and
column count is what drives TabFM's cost, so this both speeds up and improves the fit. A raw
string column inherits the score of its own encodings. TabFM then runs with
`norm_methods = ['none', 'power', 'quantile', 'robust']` and the predictions are returned
unmodified.

## build.sh

There is nothing to build. `final/build.sh` is a no-op that exits 0 with a comment saying so.
The pipeline reads no cache, no precomputed feature file and no auxiliary artifact — every
statistic is recomputed inside `engineer()` from the fold's own training rows, which is also
what keeps the target encodings honest. It needs no time and no disk.

To reproduce:

```
cd <this workspace> && bash final/build.sh && python final/run.py
```

## Things a reader could get wrong

- **The encodings are fold-local by design, not by omission.** It would be natural to
  precompute the count/target maps once over the whole training file and cache them. Doing so
  leaks labels across the CV boundary and makes the validation numbers meaningless. The
  cross-fitting inside `engineer()` is the method, not an implementation detail.
- **`N_KEEP = 200` is tuned against the *clean* encodings.** An earlier value of 220 was tuned
  while the encodings still had the context/query mismatch; after that was fixed, 180/200/220
  scored 0.1503 / 0.1501 / 0.1517. Re-tune it if the feature set changes.
- **`_as_str` exists for a real reason.** Arrow-backed string dtypes do not survive the pickle
  the judge uses to hand data to the hooks on every server, so string columns are forced to
  plain `object` dtype (and NaN to the literal `"__NA__"` level) before anything else happens.
  The same defensive cast is applied to `y_train` in `preprocess`.
- Rejected variants are recorded in the module docstring of `final/pipeline.py` with their
  scores — worth reading before re-trying one of them.