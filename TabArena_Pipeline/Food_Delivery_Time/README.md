# Food_Delivery_Time — TabFM over four quantities, encoded as steps

## What the pipeline does

The task is to predict `Time_taken(min)` (RMSE, lower better) for 15,151 deliveries from 30,300
training rows. The pipeline's premise, established from the training data alone, is that almost none
of the 11 raw columns carry signal. The target is `5*bucket + Uniform{0..4}` — `y % 5` is exactly
uniform, so roughly 2.0 of the 87.7 total variance is irreducible label noise and the only
predictable object is the bucket. The geography collapses too: `Delivery_location_lat/lon` equals
`Restaurant_lat/lon` plus a delta that is identical in both axes to 1e-6 and takes only twelve
distinct values, always positive, so direction is never recoverable and the whole four-column
coordinate block reduces to that one quantised scalar — which beats both the haversine distance it
implies and the raw columns, partly because 8% of rows have (0,0) coordinates where the delta
survives and a distance does not. Absolute position, the city prefix, `Type_of_order`, the 1320-level
delivery-person ID, out-of-fold target encodings of restaurant/city/person, and every unsupervised
count tried (rows per person, per restaurant, per coordinate; deliverers per restaurant; distance
from the city centroid) all scored at or below the base. Row order carries nothing.

That leaves four quantities — courier age, courier rating, the delta, and vehicle type — and each
acts as a **step, not a slope**: age <30 averages ~23 min against ~29.4 for >=30 and is flat inside
each plateau; rating splits at 3.95 and 4.45; motorcycle/bicycle ~27.6 against
scooter/electric_scooter ~24.4. So each is handed to TabFM in a mix of views — the raw numeric value,
an unordered string (for rating and delta, whose levels are not monotone: rating 5.0 is slower than
4.8, delta 0.04 slower than 0.05), and an explicit step code — because the normalisation TabFM
applies to numeric columns before attending smears an edge that a code states outright. Age gets no
string view: it is flat within each plateau, so its 22 levels were 22 ways to overfit. On top sits one
coarse conjunction, `cell` (48 levels, ~630 rows each), of the four step codes. `cell` also carries
the data-quality repair: the ~1450 rows with `age==29 & rating==4.6` are a median-imputed courier
record (their overlap is 3x what independence predicts, and they average the unconditional mean of
25.7 rather than the 21.6 of the cell they would otherwise land in), and age 15 / age 50 are
sentinels; all of these get their own `'u'` level instead of being read as young-and-highly-rated.
A separate `both` flag marks the imputed pair for the model directly.

The prediction itself is a single frozen TabFM pass over these nine columns with
`permute_categorical=True` and `enable_nnls=True`; no blending, no shrinkage, no auxiliary model.
Those were all tried and rejected: an affine shrink probe puts the optimum at 1.0006, HistGBR and
ridge-on-interactions errors correlate at 0.988 with a best blend weight of 0, and the 102 exactly
duplicated feature rows bound the within-duplicate variance at 60.2 (95% CI [47, 79]) against the
pipeline's MSE of 51.4 — there is no feature left to find. Local 3-fold CV RMSE 7.16854; test 7.17355,
2.0% below the published TabFM-Ensemble baseline of 7.313547 on these folds and better than both
baselines.

## What build.sh builds

Nothing — it prints one line and exits 0. `pipeline.py` opens no file and names no cache; every
column it uses is computed inside `_feat()` from the raw dataframes the judge passes in. There is
no `final/build/` either: the run left no helper script behind, because all the analysis behind the
constants above was done in throwaway shell heredocs and its conclusions are the constants. Runtime
is instant, disk is zero.

The full reproduction is still the documented two commands:

    cd <this workspace> && bash final/build.sh && python final/run.py

`run.py` needs the competition data under `data/` and a GPU for TabFM. The scored run took ~209 s end
to end (about 76 s fit, 120 s predict across 9 TabArena folds).

## Things a reader would otherwise get wrong

- **`.aux_pred.npz` in the workspace root is dead.** It holds auxiliary HistGBR predictions from the
  rejected blend at evals 12/13/50; the machinery was removed at eval 87 and `pipeline_093` never
  reads it. `build.sh` deliberately does not regenerate it.
- **`delta` is a rounded scalar, not a distance.** `_feat` averages the latitude and longitude deltas
  (they agree to 1e-6) and rounds to 2 dp to kill float noise. The rounding is load-bearing: it is
  what makes the twelve levels line up, and what makes `delta_s` a usable 12-level string.
- **`delta_bucket` is not monotone by design.** Its `np.select` maps the 0.045–0.065 band back to
  code `1`, because delta 0.04 (25.5 min) sits nine sigma above 0.05/0.06 (23.9). This looks like a
  typo and is not.
- **The step thresholds and the `'u'` markers are fitted to this training set**, not general domain
  facts. On different data the plateau edges (3.95/4.45, age 30) and the imputation signature
  (29 / 4.6, 15, 50) would have to be re-derived.
- **The docstring LOG is the real experiment record**, including the rejected branches, and the
  history of what produced it is in `history/pipeline_0NN.py` and `log.txt`. Deltas after eval 2 are
  in the third decimal place; the fold-to-fold spread is much larger than most of them, so no single
  late line should be read as significant on its own.