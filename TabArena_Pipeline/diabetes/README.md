# diabetes — final pipeline

## What it does

The task is the Pima diabetes table (512 training rows, 8 numeric features, binary target,
metric `1-auc`, lower is better). TabFM makes every prediction; the pipeline only decides what
it sees and how its ensemble is configured.

Three things happen around the frozen model. First, the physiologically impossible zeros in
`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin` and `BMI` are turned back into `NaN`, so
TabFM reads them as missing rather than as real measurements at the bottom of each range.
Second, a label-noise filter drops the 10% of training rows whose own label agrees least with
their measurements: an out-of-fold logistic regression (10-fold, on the engineered feature view)
scores every training row, each row's agreement is the predicted probability of its own class,
and the worst decile is removed before TabFM ever sees the table. This runs on the training
split only — the test rows are never filtered. Third, the feature view is widened with a small
hand-picked set of ratios and transforms (age caps at 40/50, `BloodPressure`/`Age` and its
inverse, `Pregnancies`/`Age`, `SkinThickness`/`DiabetesPedigreeFunction`, `Insulin`/`Age`,
`sqrt(Pregnancies)`, `log(DiabetesPedigreeFunction)`, `log(Glucose)`, `sqrt(Insulin)`), with
infinities mapped to `NaN`. The same engineered view feeds both the noise filter and TabFM.
TabFM itself runs with per-member probability averaging (not logit averaging), no class shift,
`sqrt`-many feature crosses and SVD features, power normalisation, and NNLS ensemble weighting
at `beta=1.0`. `postprocess` is the identity. `folds()` replaces the judge's plain 3-fold CV
with 3-fold repeated 12 times (36 scored folds) — that was purely to make the search's own
comparisons readable on 512 rows; it does not change the model.

## build.sh

There is nothing to build. `pipeline.py` reads no files: all four hooks operate on the
DataFrames the judge passes in, and `TABFM_KWARGS` is a literal. `final/build.sh` therefore
prints a line and exits 0 — it takes no time and no disk.

The scripts under `final/build/` are the run's offline analysis tools, not build steps, and
`build.sh` deliberately does not call them. `off.py` loads per-fold `.npz` dumps that a
temporary debug hook (in an earlier, discarded `pipeline.py`) wrote under `dump/`; `fsearch.py`
does greedy forward feature selection on top of those dumps with a cheap logistic-regression
proxy — that search is where the ratio features above came from; `honest.py` re-runs the same
selection on one half of the rows and measures whether it transfers to the other half; `cmp.py`
reads `log.txt` and does a paired comparison between two past evaluations. Reproducing them
would require regenerating the `dump/` directory with the old debug hook, which is not part of
the scored pipeline.

## Things a reader could get wrong

- The row filter is a training-side operation inside `preprocess`; the 10% figure is a fixed
  fraction of the rows the pipeline is given, not a probability threshold.
- `_ratios` is applied twice: once inside `_clean` to build the logistic regression's design
  matrix (with `row_id` dropped), and once in `engineer` for TabFM. Changing it changes both.
- `N_REPEATS = 12` only affects validation-time folds. The final scoring uses the harness's own
  splits, so removing it would not change the predictions, but it does change `folds_sig` and
  hence which past evaluations are comparable.
- Run it with `python final/run.py` from the workspace root; `run.py` copies `final/pipeline.py`
  over `pipeline.py` and refuses to proceed if the workspace copy has been edited to something
  different.