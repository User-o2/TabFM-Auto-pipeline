# Fitness_Club — final pipeline

## What it does

The task is to predict `attended` from six columns of 1000 gym-signup rows, scored as
1 − ROC-AUC. Almost all of the signal sits in `months_as_member`, which is monotone in the
attendance rate over every decile; ranking rows by that raw column alone already beats the
untouched TabFM pipeline. The method here is therefore about feeding TabFM a clean version of
that structure rather than about modelling capacity.

Three columns — `day_of_week`, `weight`, `days_before` — are dropped, each of which hurt TabFM
on its own. One column is added: `prior`, the decision function of a heavily shrunk
standardised logistic regression (C = 0.05) on the surviving columns, computed out of fold
across 5 stratified folds for the training rows and from a full fit for the test rows. That
single column is the largest gain of the run (0.17136 → 0.16844). Its out-of-fold noise is
part of why it helps: smoothing it by averaging several out-of-fold splits is markedly worse.
TabFM then predicts by in-context learning over those five columns, with `quantile` as its
first normalisation (the target is a ranking and `months_as_member` is heavily skewed) and
`permute_categorical` on for extra per-member diversity across the two remaining string
columns. `postprocess` returns TabFM's probabilities unchanged.

Validation uses 5 repeats of stratified 3-fold — 667 train / 333 validation, matching the 2:1
ratio the final split uses — which puts the paired resolution between two runs at about 0.0006.
Progress over the run: 0.17374 (identity) → 0.16844.

## What `build.sh` builds

Nothing. `final/pipeline.py` reads no files; the `prior` column is fitted inside `engineer()`
from the frames the judge passes in. `build.sh` is a no-op that exits 0 — zero runtime, zero
disk. Run:

```
bash final/build.sh && python final/run.py
```

`python final/run.py` fits and scores in a couple of minutes on the competition data; the TabFM
pass is the only cost.

## Things a reader could get wrong

- **`final/build/runexp.py` is not a build step.** It is the search driver used during the run:
  it rewrote the constants at the top of `pipeline.py` (`TABFM_KWARGS`, `DROP`, `PRIOR_C`,
  `PRIOR_K`, and two constants that no longer exist in the final file) and shelled out to
  `judge.py check` / `judge.py val` for each variant. Running it now would *overwrite*
  `pipeline.py`. It is kept only as a record of how the numbers in the docstring were produced.
- **The `prior` column is refitted, not restored.** It depends on which rows are in the
  training frame, so it is different for every fold and for the final fit. That is deliberate —
  a frozen precomputed version would leak across folds and, per the run's own experiments, a
  less noisy version scores worse.
- **`engineer` skips the prior when the label slice is single-class.** `judge.py check` runs the
  hooks on a 200-row slice that may contain only one class; in that case the four base columns
  are returned without `prior`. This is a guard for the dry run, not a code path the real fit
  takes.
- **The constants encode tuned optima, not defaults.** `PRIOR_C = 0.05` sits at the bottom of a
  measured curve (0.015 → 0.1 gives .16909 / .16873 / .16844 / .16883 / .16904), and the drop
  list is column-by-column measured. Changing them is not free.