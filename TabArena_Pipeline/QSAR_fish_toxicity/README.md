# QSAR fish toxicity — pipeline_094

604 training rows, 6 numeric molecular descriptors, target `LC50`, scored by RMSE (lower is better).

## What the pipeline does

TabFM remains the predictor; everything around it is a small amount of local-neighbourhood
regularisation, which is what a dataset this size and this smooth responds to.

- **Target smoothing (`preprocess`).** An out-of-fold estimate of `LC50` is built from the standardised
  six descriptors by averaging three cheap learners — distance-weighted 5-NN, an RBF `SVR(C=3)`, and a
  200-tree `ExtraTreesRegressor` — under a 10-fold split. TabFM is then trained on
  `0.70 * y + 0.30 * oof`: label noise on individual compounds gets pulled toward what their neighbours
  say, which is worth more than it costs in bias at n=604.
- **Feature engineering (`engineer`).** Six ratio/product crosses of the four continuous descriptors
  (`MLOGP`, `SM1_Dz(Z)`, `CIC0`, `GATS1i`) are appended: `m/g`, `s/g`, `c/g`, `m*c`, `c*s`, `m*s`. The
  two count features (`NdsCH`, `NdssC`) are left alone. `engineer` also pickles the original six
  columns and the raw targets to `.pipe_state.pkl`, because `postprocess` runs in a separate process
  and the workspace is the only channel between them.
- **TabFM knobs.** `n_feature_crosses='sqrt'`, `outlier_threshold=2.0`, `feat_shuffle_method='none'`.
- **kNN blend (`postprocess`).** A distance-weighted 10-NN fitted on all training rows (raw targets,
  standardised descriptors) predicts the test rows, and the final answer is
  `0.74 * tabfm + 0.26 * knn`. TabFM still carries the prediction — `tabfm_corr` on the test pass was
  0.998.
- **Validation (`folds`).** 3× repeated 5-fold CV, stratified on target deciles, so every row is scored
  three times. A single 5-fold split on 604 rows moves by ~0.002 on fold assignment alone, which is the
  same size as the effects being selected between.

Held-out test RMSE 0.8483 (CV selection score 0.8329). Over the 30 TabArena folds it averaged 0.8460
against 0.8535 for plain TabFM and 0.8539 for the TabFM ensemble baseline.

## What build.sh builds

Nothing. The pipeline reads no cache it does not write itself (`.pipe_state.pkl` is produced by
`engineer` on every pass), and the run created no helper scripts, so `final/build.sh` is a comment and
`exit 0`. It takes no time and no disk. There is no `final/build/` directory for the same reason.

    bash final/build.sh && python final/run.py

## Things a reader could get wrong

- **`.pipe_state.pkl` is written into the workspace root, not `final/`.** `run.py` chdirs to the
  workspace before calling the hooks, so the relative path resolves there. If two runs of this pipeline
  ever share a working directory concurrently they would clobber each other's state file; sequential
  runs are fine because `engineer` always rewrites it before `postprocess` reads it.
- **The 0.30 target smoothing is a training-side transform only.** `preprocess` returns a modified
  `y_train`; the metric is still computed against the untouched labels, and `postprocess` blends in a
  kNN fitted on the *raw* targets, which partly undoes the shrinkage on the prediction side. The two
  weights (0.30 and 0.26) were tuned together — moving one without the other is not a small change.
- **`_add` uses `eval` over a fixed table of expressions.** That is cosmetic, not dynamic: `NEW` is a
  module-level literal, so the feature set is static.
- **`_knn_oof` is dead code** left over from an earlier candidate; nothing calls it.
- Total runtime is dominated by the TabFM pass; the sklearn parts are seconds on this data (the
  recorded 30-fold run spent ~90 s in `pre` and ~30 s in `post` across all 30 folds).