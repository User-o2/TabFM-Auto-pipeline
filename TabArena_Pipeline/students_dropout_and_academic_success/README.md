# students_dropout_and_academic_success — final pipeline

Metric: multiclass log loss, lower is better. 2949 training rows × 1475 test rows per fold,
8 TabFM estimators. Scored test log loss **0.511122** (mean over the 9 TabArena folds; the
published TabFM baseline is 0.50716 plain / 0.505215 ensemble).

## What the pipeline does

Frozen TabFM does all the predicting; the pipeline gives it two things. First, **external
ordinal structure**. Four of the columns — `Mothers_qualification`, `Fathers_qualification`,
`Mothers_occupation`, `Fathers_occupation` — are ladders (schooling levels, ISCO-style skill
levels) that TabFM's categorical encoder turns into unrelated integers, and most of the ~28
maternal levels appear only a handful of times in a 2200-row context, so those rows can borrow
strength from nobody. `engineer` maps each level through a regex rule table written from outside
knowledge and adds a numeric rank column beside the untouched string column. The ranks are
*evenly spaced positions on the ladder*, not years of schooling: under `'none'` normalisation the
raw gaps are what the model sees, and 16→18→21 at the top against 8→8.5→9 in the middle asserts
a distance the data does not support. Levels matching no rule (`Unknown`, `(blank)`,
`Other Situation`, and parents recorded as students) stay NaN, which TabFM handles natively.
Second, **no synthetic features at all** — feature crosses, SVD components, curricular-unit
ratios and extra normalisations each measured as a loss, while transformations that only permute
what is already there pay off. So `preprocess` and `postprocess` are identity, and the remaining
gains come from TabFM's own knobs: NNLS weighting of the 8 members on their out-of-fold
probabilities plus vector calibration (a per-class scale and bias) on the same probabilities.
`folds()` overrides the judge's 3-fold CV with 2 repeats of stratified 4-fold, which scores every
row twice and puts the context size (2212 rows) nearer the 2949 the final fits on.

## build.sh

Nothing to build — `bash final/build.sh` prints nothing and exits 0 in well under a second, with
no disk cost. The pipeline reads no cache, no embedding and no fitted artifact; the rank tables
are inline constants in `final/pipeline.py`. Run `python final/run.py` directly.

## Things a reader could get wrong

- `final/build/dtcheck.py` is the only script this run wrote besides the pipeline, and it is a
  **guard, not a build step**: it asserts that no hook output carries pandas `StringDtype` (the
  judge's interpreter cannot unpickle it) and writes no files. It must be run from the workspace
  root, not from `final/`.
- `TABFM_KWARGS` sets `average_logits: False` because `enable_nnls: True` requires it. The two
  are not independent knobs.
- The regex rule tables are order-sensitive — first match wins, so the specific patterns must
  stay above the general ones (e.g. `12th year of schooling - not completed` before
  `12th year of schooling or eq`). Reordering them silently changes the encoding.
- `folds()` affects only the search-time CV. Changing it does not change what `run.py` produces,
  but every effect size quoted in the pipeline docstring was measured on those folds and is not
  comparable to a judge 3-fold number.
- `run.py` copies `final/pipeline.py` over the workspace `pipeline.py` and refuses to run if the
  two differ, so leave the workspace copy as it is.