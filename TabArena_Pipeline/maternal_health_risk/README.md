# maternal_health_risk — final pipeline

## What it does

676 training rows, six numeric columns (Age, SystolicBP, DiastolicBP, BS, BodyTemp, HeartRate),
three risk classes, scored by log loss. The predictor is a stock frozen TabFM with no ensemble
knobs set (`TABFM_KWARGS = {}`); all the work is in the feature representation handed to it. Four
derived circulatory axes are added — pulse pressure (SBP−DBP), shock index (HR/SBP), the SBP/DBP
ratio, and fractional pulse pressure (pulse pressure over mean arterial pressure) — the two rows
whose HeartRate reads 7 bpm are blanked to NaN, and the raw DiastolicBP column is dropped because
it is exactly recoverable from SystolicBP plus the derived axes and only dilutes them. Nothing is
done after TabFM: it comes out calibrated (the best global temperature is 1.05, worth 0.0007), and
blending in RF / ExtraTrees / kNN / logistic out-of-fold predictions, or perturbing the context
rows in any way, measured worse. The dataset's defining property is that 131 of 676 rows sit in
groups sharing a feature vector with disagreeing labels; those disagreements *are* the target
probabilities, and TabFM reads them better than an explicit duplicate-group estimator, so the
pipeline leaves them alone.

`folds()` replaces the judge's default 3-fold CV, whose 0.042 fold spread on 676 rows is too coarse
to rank changes worth 0.005, with `RepeatedStratifiedKFold(10 splits × 8 repeats, seed 1234)`.
`postprocess` additionally writes each fold's raw probability matrix to `.dump/final/` so that
post-processing ideas could be scored offline (`final/build/oof.py`) instead of costing an
evaluation; it does not alter the predictions.

## What build.sh builds

Nothing. The pipeline reads no cache, feature store, or precomputed artefact — it is four pure
hooks over the DataFrames the judge supplies. `final/build.sh` is a comment and `exit 0`; it takes
no time and no disk. Go straight to:

    bash final/build.sh && python final/run.py

## Things a reader could get wrong

- **`final/build/` is not part of the build.** Those four scripts are search-time tooling:
  `precheck.py` is a stricter free dry run than `judge.py check` (it reproduces the judge's fold 0
  including the pickle round-trip and real dtypes), `drive.py` and `drive2.py` rewrite module-level
  constants of `pipeline.py` and invoke `judge.py val` so several configurations run from one shell
  call, and `oof.py` rescores the dumped fold probabilities offline. They also expect to live in the
  workspace root next to `pipeline.py` and `judge.py`, not in `final/build/`, so they will not run
  in place without being copied up.
- **`DUMP` has a side effect.** Running the pipeline creates `.dump/final/*.npz` (a few hundred KB
  at most). It is wrapped in a bare `except: pass`, so a read-only workspace is harmless. Set
  `DUMP = ''` to switch it off; predictions are identical either way.
- **The dropped column is not lost information.** `DiastolicBP` is removed only after PulsePressure
  and friends are computed from it; every norm_method standard-scales each column, so an affine
  copy of an existing column is invisible to TabFM and a genuinely redundant one mildly hurts.
- **The score margin is small and the ordering inside the winning family is noise.** On the
  selection split this pipeline scores 0.37331 log loss against 0.37941 for the identity pipeline;
  the eight runners-up lie within one paired standard error (0.0013) of it. The −0.005 from the
  derived-axis family is real, the choice of exactly these four axes was decided by simplicity as
  much as by the number.