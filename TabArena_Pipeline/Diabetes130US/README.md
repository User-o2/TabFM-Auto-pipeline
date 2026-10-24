# Diabetes130US — 30-day readmission (metric: 1 − ROC-AUC, lower is better)

## What the pipeline does

TabFM makes the prediction on its own, in one frozen in-context pass over the whole training
split — there is no `sample()` hook, `postprocess` is the identity, and no other model is blended
in (`tabfm_corr` = 1.000). All the work is in how the table is presented to it, plus two ensemble
knobs.

The feature step is domain rewriting of the hospital-encounter table, applied identically to the
context rows and the query rows: the four constant / near-constant diabetes-drug columns are
dropped; `age` and `weight` buckets become numeric midpoints (and the raw bucket strings are
dropped — keeping both was measurably worse); each of `diag_1/2/3` gains an ICD-9 chapter label
(with 250.x split out as its own "Diabetes" chapter, V- and E-codes as their own levels) and a
numeric code, while the raw code string is kept at full cardinality (~670 levels); the 24
discharge dispositions collapse into 7 clinical buckets alongside the raw column; and a handful of
counts are derived — prior-visit utilisation, diabetes/circulatory diagnosis counts, medications
taken / raised / lowered / changed, and per-day intensity ratios for medications, labs and
procedures. `medical_specialty` and `admission_source_id` are mapped through a frequency lump
whose threshold is 1, i.e. every level seen in the fold's training rows survives and only levels
unseen there fall to `'rare'`; that map is built inside `engineer` from the training rows of the
current fold only, so no label and no test information leaks. Missing values are left as NaN for
the plain categoricals and carry an explicit `'NA'` level only where "not recorded" is itself
informative. TabFM is then run with `norm_methods=['none','quantile_rtdl']` (the eight ensemble
members split across two normalisations) and `outlier_threshold=20.0` instead of the default 4.0.

`folds()` overrides the judge's 3-fold CV with a 5-fold stratified split. That was the pivotal
choice of the run: at 3 folds each context is 31.8k rows, well short of the 47.7k the final model
actually fits on, and several conclusions reached there — pruning weak columns, keeping the
`power` normalisation, lumping rare ICD codes — all reversed once measured at 5 folds. The
last 3-fold configuration scored 0.33024; the shipped one scores 0.32708 at 5 folds.

## What `build.sh` builds

Nothing. It is a comment and `exit 0`. The pipeline reads no cached artifact — every table it
uses is derived inside the hooks from the DataFrames the judge passes in, and every lookup
(drug list, bucket midpoints, ICD chapter ranges, discharge buckets) is a literal in
`final/pipeline.py`. Zero runtime, zero disk.

To reproduce:

```
cd <this workspace> && bash final/build.sh && python final/run.py
```

`final/run.py` writes `final/submission.csv`. Runtime is dominated by the single TabFM pass over
~48k context rows on a GPU; there is no training loop.

## Things a reader could get wrong

- **The docstring in `pipeline.py` is a run log, not a spec.** It narrates 71 evaluations in
  chronological order, and several of its earlier conclusions are explicitly overturned later in
  the same docstring (evals 60–71). Only the code below it, and the final section of the
  docstring, describe what actually ships.
- **`WEAK = []` and `MIN_DIAG = 1` / `LUMP = {...: 1}` are deliberate no-ops**, not leftovers.
  They are the knobs whose non-trivial values won at 3 folds and lost at 5; the empty list and the
  threshold of 1 are the *result* of that re-measurement. `DEAD` (four drug columns) is the only
  column prune that survived.
- **`folds()` changes the validation protocol, not the model.** The final fit uses the whole
  training set; `folds()` only affects `judge.py val`. Scores in `log.txt` before eval 60 are on a
  3-fold split and are not comparable to the ones after.
- **`pd.set_option('mode.string_storage', 'python')` at import, and the object-dtype coercion at
  the end of `engineer`, are required, not stylistic.** The judge runs the pipeline in a separate
  process and cannot unpickle arrow-backed string arrays; removing either makes the run fail
  rather than merely differ.
- **Rows are exchangeable here** — no time or group structure was found in the encounter table, so
  a plain stratified split mirrors how the test split was made.