# hazelnut-spread-contaminant-detection — final pipeline

## What it does

The task is 1600 training rows × 30 numeric sensor readings (`s<i><j>`, the off-diagonal of a 6×6
transmission matrix between six sensors), scored as 1−AUC, lower better. The pipeline hands that
table to TabFM essentially untouched and spends its whole budget on TabFM's own ensembling.

`preprocess` and `postprocess` are the identity: the metric is rank-based, so TabFM's raw output is
already the thing being scored. `engineer` adds no columns — it only applies a fixed random
permutation of the 30 sensor columns (`COLORDER = 'rand4'`, seeded, so train and test get the same
order) and normalises the column index to plain Python objects so the frames survive being pickled
back to the judge. The permutation matters only because TabFM's feature-cross generator is
order-sensitive; it is a seed choice, not a transformation.

The tuning is all in `TABFM_KWARGS`: eight ensemble members spread over five normalisations
(`none`, `power`, `quantile`, `quantile_rtdl`, `robust`) with `n_feature_crosses='sqrt'` and
`outlier_threshold=3.0`. `none` and `power` carry the accuracy; the other three are there purely as
diversity. Widening the normalisation list from TabFM's default two and turning crosses on is the
whole measured gain (1−AUC 0.00160 → 0.00152 over 50 validation splits).

Two negative results are load-bearing and are documented at length in the pipeline docstring.
Every engineered representation that was tried lost — symmetric/antisymmetric pairs, row/column
sums, Frobenius norm and singular values, row normalisation, PCA rotation, duplicated columns,
mutual-information column selection — and so did every attempt to drop or synthesise rows. And
classical models plateau around 1−AUC 0.025 while TabFM sits at 0.0016, so there is nothing worth
stacking on top of it.

`folds()` overrides the judge's default 3-fold with 10-fold × 5 repeats (`RepeatedStratifiedKFold`,
`random_state=1`). The effects being chased here are ~1e-4 while 3 folds spread by ~4e-4; 50 splits
bring the paired standard error to ~5e-5, and 1440 training rows per split also mirror the 1600 the
final model fits on. The 2400 rows contain no exact duplicates and no replicate clusters, so plain
stratified k-fold is the right mirror of how the task's folds are cut.

## What build.sh builds

Nothing. `final/build.sh` is a documented no-op that exits 0. The pipeline opens no files, loads no
caches and fits no auxiliary model, so there is no artifact standing between `data/train.csv` and
the TabFM pass. It runs in well under a second and needs no disk.

```
cd <this workspace> && bash final/build.sh && python final/run.py
```

## Things a reader might otherwise get wrong

- **`final/build/` is not part of the build.** `sweep.py` (rewrite a module constant in
  `pipeline.py`, then run `judge.py val`) and `cmp.py` (paired fold-by-fold comparison of past
  `log.txt` results) were the search loop's driving and analysis tools. They produce no input to
  `pipeline.py` and `build.sh` deliberately does not call them.
- **`COLORDER` is not dead code.** It looks like a no-op reshuffle, and it is one as far as the raw
  features go, but TabFM's feature-cross generator consults column order, so changing the seed
  changes the prediction.
- **`feat_shuffle_method` is inert here.** Whenever feature crosses are enabled, that branch of
  TabFM's `EnsembleGenerator` builds its own permutations and never reads the flag — which is why
  member diversity had to come from the normalisation list instead.
- **The scored numbers in the docstring are 50-split validation means**, not test scores, and the
  differences quoted (1e-4 scale) are only meaningful against the ~5e-5 paired standard error that
  the 10×5 fold scheme buys. Re-running with the judge's default 3 folds will not resolve them.
- `run.py` copies `final/pipeline.py` over the workspace `pipeline.py` before running, and refuses
  to do so if the two differ — they are identical as shipped.