# houses — TabFM pipeline (pipeline_093)

## What it does

The task is the California block-group housing data: predict `LnMedianHouseValue` from eight columns,
scored by RMSE on the log scale. TabFM makes every prediction; the pipeline only changes what it sees.

Two edits to the table, both geometric. First a **reparameterisation**: the four collinear size columns
(`TotalRooms`, `TotalBedrooms`, `Population`, `Households`) become one level (`Households`) plus three
ratios — rooms per household, bedrooms per room, people per household — rather than being appended
alongside the originals. Second, **neighbourhood geometry**, which is the part TabFM cannot construct
for itself from two coordinates:

- a **density profile**: the great-circle distance to the *k*-th nearest block group for ten scales
  *k* = 2, 3, 6, 10, 20, 30, 60, 100, 200, 300, computed on a KD-tree over the **pooled train + test**
  coordinates (features only — no test target is ever touched), self-matches excluded. Ten scales
  handed over whole beat any summary of them.
- **`coast_km`**: kilometres to the nearest point of a hand-written California coastline polyline, with
  the San Francisco / San Pablo / Suisun bay shoreline folded into the same reference set. Both
  polylines are densified with interpolated vertices so the minimum over points approximates a minimum
  over the line.
- **`metro1..4_km`**: distances to Los Angeles, San Francisco, San Diego and San Jose, **sorted**
  ascending, so the model gets the whole profile instead of just the nearest.

That takes 8 columns to 23. TabFM's own knobs are set to `norm_methods = [power, quantile,
quantile_rtdl, robust]` (the unnormalised `none` view is dropped — it stops paying once two thirds of
the columns are long-tailed distances) and `outlier_threshold = 8.0`. Postprocessing is a single
monotone step: clip predictions to the observed training target range, because the target is censored
at ln(500001) = 13.1224 and about 4.7% of rows sit exactly on that cap.

No target-derived neighbourhood feature is used anywhere. kNN target encoding, label smoothing toward
or away from a neighbourhood mean, and neighbourhood covariate means were all measured and all lost:
TabFM already aggregates over its context, so pre-aggregating only destroys resolution.

Scores: CV 0.17979 (identity baseline 0.18603 on the same folds, `folds_sig=f5e648f0`); final test
0.177057, mean over the 9 TabArena folds, against published TabFM baselines of 0.181622 (plain) and
0.181461 (ensemble).

## What build.sh builds

Nothing. `pipeline.py` reads no cache, no auxiliary file and no artefact of any kind — every feature is
computed in-process from the `X_train` / `X_test` frames the judge passes in, and the coastline, bay
and metro coordinates are literals in the module (hand-written, not derived from data). `build.sh`
therefore prints one line and exits 0: no runtime, no disk. `final/build/` is empty for the same
reason; the scripts written during the run (`expl.py`, `expl2.py`, `kw.py`) were exploration that
printed to the terminal and produced no file.

    cd <this workspace> && bash final/build.sh && python final/run.py

## Things a reader could get wrong

- **The density profile pools the test coordinates.** This is deliberate and worth −0.0006: it uses
  test *features* only, never test targets, and gives one density estimator instead of two slightly
  different ones. It does mean the feature values for a training row depend on which test rows are
  present, so results are not identical if you change the test split.
- **The self-exclusion in `_density` is not "drop column 0".** Co-located block groups tie at distance
  0, so the code masks the neighbour whose index equals the row's own, with a fallback for rows where
  no such tie appears in the retrieved *k*.
- **Distances are a flat-earth approximation**, `111 km/degree` with a `cos(lat)` correction on
  longitude (the KD-tree uses a fixed `cos(35.6°)` scaling). A spherical metric was tried and did not
  help; do not "fix" this expecting an improvement.
- **`metro1_km` is the old single min column** — the sorted profile is a superset, not a replacement of
  a different quantity.
- Runtime for the full 9-fold final was ~131 s wall clock at 8 estimators (≈19 s preprocess, ≈97 s
  predict) on the run's GPU; a single `python final/run.py` is a fraction of that.