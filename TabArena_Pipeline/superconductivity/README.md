# superconductivity — TabFM pipeline (RMSE, lower is better)

## What the pipeline does

Frozen zero-shot TabFM predicts `critical_temp` by in-context learning over the whole training
table; everything shipped here is about *how much variety* the eight ensemble members see, because
that turned out to be the only lever on this task that moved the score. No row is ever dropped,
resampled, deduplicated or synthesised, and no column is added, selected or rotated — a long list of
ablations in the pipeline's docstring records that each of those cost between 0.03 and 0.28 RMSE.

Four things ship:

- **Four normalisation views** (`none`, `power`, `quantile`, `robust`) spread across the eight
  members instead of TabFM's default two, with `n_svd_features="sqrt"` adding SVD columns to the
  odd-indexed half — i.e. to the `power` and `robust` views, the pairing that won a six-way sweep.
- **`log1p` on the non-negative columns with skew > 2.5** (nine of the 81 — mostly the FusionHeat
  statistics), which spread over four orders of magnitude with nearly all their mass near zero. This
  mainly helps the members that normalise with `none`, which otherwise see a spike plus a long tail
  that z-score clipping flattens.
- **Row-order draws.** TabFM is *not* permutation-invariant over the context rows, so the pipeline
  hands the judge four index orderings of the full training set (identity plus three seeded
  permutations). The draws see identical rows and identical columns, so averaging them costs no
  information and cannot raise the expected error. Measured, the gain is ~0.001 — inside the noise;
  it ships on the argument, not the measurement.
- **A second target parameterisation.** The judge fits one TabFM per column of `y_train`, so
  `preprocess` passes `[y, sqrt(y)]`; `postprocess` squares the second prediction back and blends at
  `0.17`, then clips to the training target range. The two fits correlate 0.989, below the
  `sigma_raw/sigma_sqrt = 0.993` break-even, so a positive weight is a priori right; measured, it is
  worth ~0.003.

Scored result: **8.7946 mean RMSE over the 9 TabArena folds**, against 8.9517 for the plain TabFM
baseline and 8.9176 for the ensemble baseline (search CV: 9.35488).

## What build.sh builds

Nothing. `pipeline.py` reads no files — it only transforms the DataFrames the judge passes to its
four hooks — so `final/build.sh` is a comment and `exit 0`. It runs in under a second and needs no
disk. `final/build/` is empty for the same reason: the run wrote no helper scripts, every experiment
was an edit to `pipeline.py` followed by `python judge.py val`.

    bash final/build.sh && python final/run.py     # writes final/submission.csv

## Things a reader could get wrong

- **`_DRAWS = 4`, not 8.** The docstring's long argument for row-order averaging was written when
  K=8 was current; eval 83 asked whether K=4 buys the same answer for half the compute, and 4 is what
  was scored on the test split. The K-curve is noise-dominated (9.3572–9.3598 across K=1, 3, 8, 16),
  so this is a runtime choice, not a quality regression.
- **`_W_CBRT = 0.00`.** The cbrt target column is wired up but switched off — evals 79–81 found it
  too collinear with sqrt to add a third direction. Setting it non-zero costs another full TabFM fit.
- **No `max_num_rows`.** Deliberate: a row cap would mean 9450 context rows in 3-fold CV but 14175 in
  the final, i.e. a different fraction of the data in the two places. The table is small enough to be
  used whole under the judge's 65536-row default.
- **Runtime is dominated by prediction, not fitting.** The scored final took ~37 min for 9 folds:
  ~167 s of fit against ~2014 s of predict, since each of the 4 draws × 8 members re-attends over
  the full context for every test chunk. Halving `_DRAWS` roughly halves the wall clock.
- **The numbers quoted in the docstring's ablation list are CV scores** from the 3-fold search on
  fold 0's training split, on a ~9.36 scale. They are not comparable to the 8.79 test figure above,
  which is the mean over 9 TabArena folds with more training rows.