# Another-Dataset-on-used-Fiat-500 — used Fiat 500 price, RMSE (lower is better)

## What the pipeline does

TabFM predicts the price directly from the seven raw columns: no engineered feature, no dropped
column, and default ensemble knobs (`TABFM_KWARGS = {}`). Seventy-one evaluations looked for
something better on that front and found nothing — every added column scored 723–753 against 720
for the untouched table, every dropped column 723–756, and every TabFM knob 721–725.

The one change that does help is a *density-gated blend*. TabFM's out-of-fold error is not uniform:
about 446 on rows that sit right next to a training row and about 841 on rows in sparse regions of
feature space. A plain gradient-boosted tree is much worse overall (756 vs 722) but is *not* worse
than TabFM out in that empty space. So the two predictions are averaged with a weight that is zero
on dense rows and opens up on sparse ones:

    w = 0.55 * min(d / d_ref, 1) * min(km / (2 * median km), 1)
    prediction = (1 - w) * TabFM + w * GBDT

`d` is the query row's distance to its nearest training row and `d_ref` the median nearest-neighbour
spacing of the training rows themselves, both measured on a quantile-normalised numeric view of the
fold's own training split — so the rule rescales itself when the final fits on 1025 rows instead of
the 820 a CV fold sees, rather than carrying a hard-coded distance. The mileage factor is the second
half of the gate: after conditioning on density, low-km cars still preferred `w = 0` and high-km cars
`w ≈ 0.55`. Both models work on `sqrt(price)`; `postprocess` squares the blend back. Validation is 3
repeats of 5-fold (15 folds) — at a single 5-fold the whole candidate field sat inside 720–724 and
nothing was readable.

TabFM carries the prediction on every row and essentially all of it on the dense ones
(`tabfm_corr = 0.998`).

Scored result: **693.49 RMSE**, mean over the 30 TabArena folds, against published TabFM baselines
of 703.64 (plain) and 703.70 (ensemble). Search CV was 702.69.

## What build.sh builds

Nothing. `final/build.sh` is a comment and `exit 0`.

The pipeline reads no precomputed file. `.blend.npy` and `.wblend.npy` are the only filenames it
mentions, and it writes both itself in `preprocess()` on every fold to hand the GBDT prediction and
the blend weights across to `postprocess()` — per-fold scratch, not a cache. Everything else comes
from the competition data the judge loads and passes into the hooks. Runtime and disk for the build
step are therefore zero.

`python final/run.py` (one TabFM pass, 8 estimators, on this task's ~1k train / ~500 test rows)
takes well under a minute on a GPU node; the recorded final run was 83 s of wall time across all 30
folds.

## Things a reader could get wrong

- **`final/build/` is not part of the build.** `sw.py`, `fe.py` and `kw.py` are search helpers that
  rewrite a module-level constant plus the `INTENT` comment in `pipeline.py` and then shell out to
  `judge.py check` + `judge.py val`; `explore.py` is a local sklearn-only proxy CV used to try ideas
  without spending judge budget. They are kept for the record. None of them produces an artefact,
  and running one would *edit* `pipeline.py`, not build anything.
- **The blend weights are recomputed per fold, on purpose.** `_weights()` fits its
  `QuantileTransformer` and `NearestNeighbors` on the training rows it is handed. Precomputing them
  once from the full training set would leak across folds and would also use the wrong `d_ref`.
- **`W_MAX`, `D0` and `KMF` are on a flat basin.** `W_MAX` 0.5/0.55/0.6/0.7/0.8 scored
  703.87/703.79/703.84/704.35/705.42, and the mileage scale 1.5/2.0/2.5 gave 703.97/703.84/704.48.
  The exact constants are not delicate; the shape of the gate is what matters. They were also chosen
  in a nested offline check (fit on four folds, scored on the fifth) before being measured here.
- **`folds()` is 15 folds, not the judge's default 3.** That is deliberate and it is what the
  reported search numbers (`folds_sig ac149037`) were measured on; a 3-fold rerun will not reproduce
  702.69.
- **The `row_id` column is dropped for the helper models only** (`_design`). TabFM still sees the
  table exactly as given — dropping any column from TabFM's view made things worse.