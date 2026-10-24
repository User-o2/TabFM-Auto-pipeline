# blood-transfusion-service-center — final pipeline

## What it does

The task is 748 RFMT rows (recency, frequency, monetary, time) split 498 train / 250 test, scored
as `1 - ROC-AUC`, so only the ordering of the predictions matters. The pipeline drops
`TotalBloodDonated` — it is exactly `250 * NumberOfDonations` in every row, a duplicate column —
and hands TabFM the three remaining raw counts unchanged. TabFM's ensemble is configured with a
four-way normalisation cycle (`none / power / quantile / robust`) plus both synthetic-feature
knobs (`n_svd_features='sqrt'`, `n_feature_crosses='sqrt'`); with only three columns, a random
rotation and a random cross are views no other ensemble member has, and those two knobs are where
most of the TabFM-side gain came from. Deriving RFMT ratios as *columns* consistently hurt, but the
same ratios help as *models*: alongside TabFM the pipeline fits a small fixed sklearn ensemble on
the same fold — two ExtraTrees (one entropy, one gini on the raw counts) and a
QuantileTransformer + logistic regression — on a view that adds the quotients (donation rate,
active span, mean inter-donation gap, recency in gap units). Both sides are converted to
percentile ranks and blended 60% TabFM / 40% partner; the blended rank is written back as the
positive-class probability. TabFM stays the predictor and carries the majority weight (the two
rankings correlate ~0.974, so the partner is a nudge, not a replacement).

`folds()` is `RepeatedStratifiedKFold(3 splits, 10 repeats, seed 0)` = 30 folds, mirroring the
TabArena protocol at 2/3 scale, because the judge's default single 3-fold CV had a fold spread
(std 0.056) far wider than any effect being measured.

Scored result: `1-auc = 0.24306` on the held-out test split (30 TabArena folds), against the
published TabFM baselines of 0.244169 plain / 0.241016 ensemble.

## What build.sh builds

Nothing. `final/pipeline.py` has no precomputed inputs — it reads only the competition data the
judge passes to its hooks, and the one file it names (`.partner.npy`) it writes itself as a
per-fold channel from `engineer()` to `postprocess()`. `build.sh` therefore just prints that and
exits 0: zero runtime, zero disk. There is no `final/build/` directory because the only file this
run ever wrote into the workspace was `pipeline.py`.

    bash final/build.sh && python final/run.py     # writes final/submission.csv

One end-to-end run is on the order of a few minutes on a GPU node (the scored run took 201 s
wall, of which ~105 s was preprocessing and ~33 s the TabFM forward passes, 8 estimators).

## Things a reader could get wrong

- **`.partner.npy` is scratch, not a cache.** A copy is sitting in the workspace from the last
  run. Deleting it changes nothing; leaving a stale one changes nothing either, because
  `postprocess()` compares its length to the number of prediction rows and returns TabFM's raw
  ranking unchanged if they disagree. It does mean the pipeline is **not safe to run twice
  concurrently in the same working directory** — two folds in flight would overwrite each other's
  file.
- **The blend is rank-space only.** The numbers written to the submission are percentile ranks,
  not calibrated probabilities. That is correct for AUC and wrong for anything else.
- **`sklearn` is forced single-threaded** at import time (`OMP_NUM_THREADS=1` etc.); the
  OpenMP/BLAS pools hang in the judge's sandbox. The hooks are single-threaded by design and this
  costs nothing at 498 rows.
- **The partner is a fixed mix, not a fitted one.** The weights (0.4 / 0.4 / 0.2 within the
  partner, 0.6 / 0.4 against TabFM) were chosen offline and deliberately not optimised by a greedy
  weight search — that search scored better in-sample (0.23746 → 0.23864 out-of-sample) but did
  not transfer between repeats.
- **`run.py` refuses to overwrite a differing `pipeline.py` in the workspace.** The workspace copy
  is the scored record; if the two ever diverge, reconcile them by hand rather than deleting one.