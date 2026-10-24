# online_shoppers_intention — final pipeline

Binary classification of `Revenue` over 8220 training rows x 17 features, scored as `1 - AUC`
(lower is better).

## What the pipeline does

Frozen TabFM does all the predicting, in-context over the full training set; the pipeline around it
is deliberately almost empty, because on this task nearly everything tried made it worse. The one
substantive change is feature selection: five columns are dropped before TabFM sees the table —
`OperatingSystems`, `Browser`, `Region` (near-zero importance nominal ids) and `Informational`,
`Informational_Duration` (subsumed by the Administrative counterparts). TabFM itself runs with
`n_feature_crosses='sqrt'`; no row sampling, no target transform, no post-processing of the
probabilities, no second model. The hooks keep switches for the alternatives that were measured and
rejected — `LOG1P`, `NOMINAL` (integer ids as string categoricals), `MERGE_INFO` (fold
Informational into Administrative), and `CLEAN` (drop the fraction of rows an out-of-fold
HistGradientBoosting finds least compatible with their label) — all left at their off/empty values,
since each was flat or worse against the identity baseline. The GBDT was worth 0.930 AUC against
TabFM's 0.940 and their rankings agree, so it pays neither as an extra column (+0.00441) nor as a
rank blend at the output (+0.00053); those routes are closed.

`folds()` is a repeated stratified 5-fold, 4 repeats, `random_state=12345` — 20 fits, because the
candidates that were still in contention differed by ~0.00015 and that is not resolvable at 10
folds. The rows are exchangeable (no time or group structure), so plain stratification mirrors the
test split.

## What build.sh builds

Nothing. `pipeline.py` reads no files — it only transforms the DataFrames the judge passes to its
hooks, and names no path or cache anywhere. `final/build.sh` says so and exits 0; it runs
instantly and needs no disk.

    bash final/build.sh && python final/run.py

`python final/run.py` is the real cost: it loads the competition data, runs the hooks and the frozen
TabFM pass (`judge.estimators_final` estimators, override with `-n`), and writes
`final/submission.csv`. Use `python final/run.py --check` for the hooks-only dry run with no TabFM.

## Things a reader could get wrong

- The two scripts under `final/build/` are **search tooling, not build steps**, and `build.sh`
  intentionally does not call them. `sweep.py` rewrites the `TABFM_KWARGS` line and the intent
  docstring of the workspace `pipeline.py` in place and spends judge budget running
  `judge.py val` per case; `cmp.py` only reads `log.txt` to print paired fold diffs between two
  `[RESULT n]` lines. Running `sweep.py` here would mutate the scored pipeline.
- `run.py` copies `final/pipeline.py` over `<workspace>/pipeline.py` (the judge loads it by name)
  and refuses to run if the workspace copy differs — that guard is expected, not an error to
  route around.
- The `NOMINAL` branch in `engineer` builds string columns with `dtype=object` explicitly. That is
  not stylistic: pandas 3 turns any string column into a pyarrow-backed `StringDtype`, and the
  process that unpickles the frame has no pyarrow. The branch is inactive by default but the
  constraint applies to anyone who switches it on.
- `_binary` maps the label with `a == np.unique(a)[-1]` because the target arrives as `'True'`/
  `'False'` strings from the judge but as real bools from a local `read_csv`.
- The docstring's per-experiment deltas are on the 20-fold CV, paired against the eval each one
  followed; identity there is 0.06036 and the best measured 0.06028. They are not test scores.