# MIC — myocardial infarction complications (8-class cause of death, log loss)

## What the pipeline does

TabFM is the predictor throughout; the pipeline only changes what it sees and how confident its
output is. The 1132 training rows and 111 raw columns are expanded to about 160 by three groups of
engineered features. First, the structure of the ICU chart: the per-day pain-relapse and analgesia
blocks are only filled in while the patient is still in the unit, so they are missing for a third
of a percent of the survivors and for 40–47% of the deaths; the pipeline turns that into explicit
per-day missing fractions, an `icu_days` ordinal and its products with the Killip score, the
acute-event count and age. Second, clinical composites the table has the ingredients for but not
the value — pulse pressure and MAP, the ambulance-to-ICU systolic change, infarct extent, AST/ALT,
potassium and sodium deviation from normal, age × chronic heart failure, and missingness flags for
the columns whose absence is itself informative. Third, one coarse textbook risk profile per cause
of death (shock, rupture, CHF, VF, thromboembolism, pulmonary edema, asystole), in both an integer
threshold form and a z-scored form whose means and standard deviations come from the training fold
only; predicting *which* cause accounts for roughly half the loss and the model otherwise sits near
the class prior for the rarest four causes. Two nearly empty columns are dropped and seven
roll-up sums that an ablation showed were not paying are removed after the cause scores that
consume them have been computed.

Because the judge fixes the TabFM ensemble at 8 estimators, the pipeline enlarges the ensemble in
the only remaining way: `sample()` returns seven contexts — the full training fold plus one
stratified subsample at each of 97 / 94 / 91 / 88 / 85% — and the judge averages one TabFM pass per
context. Multi-scale draws beat repeated draws at a single fraction, and pure row permutations
reproduce the single-draw score exactly (TabFM's context is order-invariant), so a draw has to give
up rows to be a different model. `TABFM_KWARGS` enables feature crosses and SVD features only.
Finally `postprocess()` raises the averaged probabilities to a global power of 0.89 and renormalises
— TabFM is mildly overconfident on this table and flattening is worth about 0.005 log loss; nothing
more elaborate (vector scaling, prior blends, per-class or row-dependent temperatures) survived.
Validation uses `folds()` = 5-fold × 3 repeats rather than the judge's shuffled 3-fold, because with
8 classes and 8–73 examples of the seven death causes the 3-fold spread swamped the effects being
measured.

## What build.sh builds

Nothing. `final/pipeline.py` reads no files — all features are computed in-process from the frames
the judge passes in — so `final/build.sh` is a comment and `exit 0`. It takes no time and no disk.

## Things a reader could otherwise get wrong

- Run it as `bash final/build.sh && python final/run.py` from this workspace. `run.py` copies
  `final/pipeline.py` over `./pipeline.py` (refusing if the workspace copy differs) and calls the
  same `judge.load_task` / `judge.run_pipeline` used during the run, so the reproduction cannot
  drift from what was scored. It writes `final/submission.csv`.
- `final/build/oof.py` is an offline analysis tool, not part of the prediction path. It rescores
  probability dumps that `pipeline.py` writes into `_oof/` — and it writes them *only if* an `_oof/`
  directory already exists in the working directory when the pipeline runs. With no `_oof/`
  directory the dump is a silent no-op, which is the intended default.
- `oof.py` also expects `.train_shuffled.csv` (the training set shuffled with `random_state=0`,
  which reproduces the judge's row order); the command that made it is quoted in `build.sh`.
- Many module-level constants in `pipeline.py` (`COLLAPSE`, `MISSING_FLAGS`, `COMPOSITES4/5`,
  `SENTINEL`, `AUX_TREES`, `OVERSAMPLE`, `ROW_SHUFFLE`, `ALIVE_ONLY`, `CONTEXT_FRAC`) are switches
  for experiments that were measured and rejected. They are off in the scored configuration, and the
  header comment records what each cost. `SCALES` overrides `DRAWS`/`DRAW_FRAC`, so editing those
  two alone changes nothing.
- Reference numbers from the run, on the 5×3-fold validation: 0.44446 for the identity pipeline on
  the judge's own 3 folds, 0.43822 for it on these 15 folds, and 0.41562 for this configuration.