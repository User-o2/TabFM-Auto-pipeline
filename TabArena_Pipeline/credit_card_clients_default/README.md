# credit_card_clients_default — final pipeline

## What it does

TabFM makes the prediction; the pipeline only changes how the credit-card table is written down for
it and how its ensemble is configured. The feature work is a single re-expression of the monetary
columns into ratios, so that a row is described by how much of its own credit line it is using rather
than by raw New Taiwan dollars: the six `BILL_AMT` columns become utilisation `BILL_AMT_i /
LIMIT_BAL`, kept as a level for the newest and oldest month and as month-over-month *changes* in
utilisation for the four in between; the six `PAY_AMT` columns become the share of the previous
month's bill that was actually paid (`PAY_AMT_i / max(BILL_AMT_{i+1}, 1)`, and against `LIMIT_BAL` for
the oldest, where no earlier bill exists), clipped at 1.75 so that one-off large repayments do not
stretch the scale for everyone else. The two categoricals get their undocumented codes folded into
the documented ones (`EDUCATION` 0 and 4+ → "other", `MARRIAGE` 0 → "other"). The `PAY_*` repayment
status columns, `LIMIT_BAL`, and the demographics pass through untouched. `preprocess` and
`postprocess` are identity; the ensemble knobs (`TABFM_KWARGS`) do the rest — sqrt feature crosses,
five different input normalisations across ensemble members (`none`, `power`, `quantile`, `robust`,
`quantile_rtdl`), class shift off, outlier clipping at 3 sigma, and averaging member **probabilities**
rather than logits. `folds()` is a plain 5-fold stratified split (shuffled, seed 0): the rows are
exchangeable, there is no time or group structure to mirror. Metric is 1 − AUC, lower is better.

## What build.sh builds

Nothing. `final/pipeline.py` reads no files — every hook is a pure function of the DataFrames the
judge passes in, with no cached features, no offline-fitted model, no auxiliary artifact.
`bash final/build.sh` prints one line and exits 0 in well under a second, and needs no disk. It
exists only so that `bash final/build.sh && python final/run.py` works as documented.

The real cost is `python final/run.py`: one frozen TabFM pass per ensemble member (8 members by
default, `judge.estimators_final`) over the full training set as in-context rows, on a GPU. That is
minutes, not hours, at this table's size (~24k rows, 23 columns), and writes `final/submission.csv`.

## Things a reader could get wrong

- **`final/build/` does not exist for this run.** The run's two helper scripts, `sweep.py` and
  `runvars.py`, were search drivers — they rewrote `pipeline.py` to a candidate variant, ran
  `judge.py check` / `judge.py val`, and printed the RESULT line. They built no data and are not part
  of reproducing the result; both were deleted at the end of the run, so there was nothing to copy.
- **The column names are reused, not added to.** `engineer` overwrites `BILL_AMT1..6` and
  `PAY_AMT1..6` in place with the derived ratios, so the frame keeps its original width and a column
  called `BILL_AMT3` in the output is a *change in utilisation*, not a bill amount. The feature count
  TabFM sees is unchanged from the raw data.
- **`BILL_AMT2..5` are differences, `BILL_AMT1` and `BILL_AMT6` are levels.** This is deliberate and
  slightly asymmetric: the loop writes `U[:, i-1] - U[:, i]` for `i` in 1..4, i.e. into `BILL_AMT2`
  through `BILL_AMT5`.
- **`average_logits: False` matters.** Averaging probabilities instead of logits across the five
  normalisation variants was one of the knobs that moved the score; it is not a default.
- `run.py` copies `final/pipeline.py` over the workspace `pipeline.py` (they are identical here) and
  calls the same `judge.load_task` / `judge.run_pipeline` used during the run, so the reproduction
  cannot drift from what was scored.