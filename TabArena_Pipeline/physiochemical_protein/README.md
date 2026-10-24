# physiochemical_protein — final pipeline

## What it does

The task is TabArena's `physiochemical_protein` regression: predict `ResidualSize` from nine
physicochemical descriptors, scored by RMSE (lower is better). TabFM does the predicting; the
pipeline is entirely about what it sees and how its output is read back.

Four things happen around the frozen model.

*Two-headed target.* `preprocess` replaces the single target with a pair of columns, `t0 = y` and
`t1 = max(y) - y`, so TabFM regresses the same quantity from both ends of its range. `postprocess`
averages the two views back together (`0.5 * (t0 + (hi - t1))`) and clips to the observed training
range. The mirrored head is a cheap second opinion whose errors are not identical to the first's.

*Feature table.* `engineer` divides three of the extensive descriptors
(`FracExposedNonPolarPart`, `MassWeightedExposedArea`, `AvgDeviationExposedArea`) by
`TotalSurfaceArea` to make size-free ratios, then drops the two raw columns whose information the
ratios carry (`AvgDeviationExposedArea`, `MassWeightedExposedArea`). The table TabFM sees is ten
columns; the search settled there — nine lost.

*Duplicate-row lookup.* Many rows in this dataset repeat exactly across the split. `engineer` builds
a lookup keyed on the nine base features rounded to 6 decimals, mapping each key to the mean training
target, and looks up every test row. Hits are blended into the final prediction at weight `W = 0.6`
against TabFM's own output. Because `engineer` and `postprocess` do not share state directly, the
hit vector is passed through `.dup_lookup.npz` in the working directory, guarded by a SHA-1 of the
test features so a stale file from another invocation is ignored rather than misapplied.

*Ensemble knobs.* `TABFM_KWARGS` runs the eight estimators over five normalisation views —
`quantile_rtdl`, `robust`, `none`, `power`, `quantile` (cycled, so the first three appear twice) —
with `max_num_rows = 65536`. `folds()` is plain 5-fold shuffled KFold: the rows are exchangeable, so
validation needs no group or time structure.

Published TabFM baselines on the same folds: plain 2.888108, TabFM-Ensemble 2.88039. The best
validation reference recorded for this pipeline's neighbourhood was 2.80260.

## What build.sh builds

Nothing. `final/build.sh` is a no-op that exits 0, because `pipeline.py` reads no precomputed file.
The one `.npz` it touches, `.dup_lookup.npz`, it writes itself during the same run. It takes no time
and no disk.

The whole run was edits to `pipeline.py` — no helper script or cache was ever written — which is why
there is no `final/build/` directory to call into.

## Things a reader could get wrong

- `.dup_lookup.npz` in the workspace root is **output**, not input. Deleting it is safe: the next run
  recreates it. If it is present but stale, the signature check makes `postprocess` fall back to the
  plain TabFM prediction instead of blending in the wrong rows.
- The lookup is not leakage. It is keyed on features and filled only from training targets; test
  labels never enter the workspace.
- `_sig` deliberately intersects with `X.columns`, because `engineer` returns a frame with two of the
  base columns dropped. Both real callers pass the raw frame, so the signature is over all nine, but
  the guard survives being called on the engineered one.
- `preprocess` returns a two-column `y`, so anything reading `y_train` after that point sees
  `t0`/`t1`, not `ResidualSize`. `postprocess` receives the original `y_orig` for the clip bounds.
- `run.py` copies `final/pipeline.py` over the workspace `pipeline.py` and refuses to run if the two
  differ; they are identical to `history/pipeline_096.py`, the scored snapshot.