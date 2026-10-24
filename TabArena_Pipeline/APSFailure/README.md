# APSFailure — scored pipeline

## What it does

The task is the Air Pressure System failure set: ~170 anonymised, non-negative, heavy-tailed sensor
counters with a lot of missing values and ~1.8% positives, scored as `1 - AUC` (lower is better).
TabFM makes the prediction; the pipeline only decides what table it sees.

Seven of the column families (`ag_*`, `ay_*`, `az_*`, `ba_*`, `cn_*`, `cs_*`, `ee_*`) are ten-bin
histograms of one underlying quantity rather than ten independent counters, so the pipeline
reparametrises each of them into the shape TabFM can actually compare row to row: a total
(`{g}_tot`, log1p'd), the ten bins renormalised to fractions of that total (`{g}_f0..f9`), and the
centre of mass of those fractions (`{g}_com`). Rows whose bins are entirely missing keep NaN rather
than a fabricated zero, and the raw bin columns are dropped — keeping both views was tried and lost
to dilution. Every remaining raw counter is log1p'd, which is the single largest win in the log, and
the number of missing values per row is added as one feature. No rows are dropped or resampled:
every attempt to subsample the negatives or bag draws scored worse on every fold, so TabFM sees the
full training set as its in-context support. The only ensemble knob touched is
`n_svd_features="sqrt"`, which hands half the ensemble members a compact SVD summary of these
strongly collinear columns. `preprocess` and `postprocess` are identity.

3-fold CV with 8 estimators put this at about 0.0057–0.0064 `1-AUC`, against 0.0071 for feeding the
raw table through unchanged.

## What build.sh builds

Nothing. `final/build.sh` is a no-op that exits 0 and says so. The pipeline reads no file from disk
— it is four pure functions of the frames the judge passes in — and no cache or intermediate
artefact was produced anywhere in this run, so `final/build/` is empty. It takes no time and no
disk. Running it is still harmless if you want to follow the documented two-step.

## Things a reader could get wrong

- `HIST_GROUPS` is a hand-picked list of prefixes, and `_build` expects the members to be named
  `{prefix}_000 … {prefix}_009`. A group with fewer than two of those columns present is skipped and
  its columns then stay in the raw log1p'd block, so the code is safe against renamed inputs but
  silently loses the reparametrisation. If you point this at a differently-named copy of the data,
  check that the histogram columns are still matching.
- `row_id` is excluded from the log1p block but is *not* dropped; the judge strips identifier columns
  itself, so removing it here is not the pipeline's job.
- The bin fractions are divided by the total only where that total is strictly positive; an all-zero
  histogram yields NaN fractions on purpose, so that "no observations" and "observations spread
  evenly" are not represented the same way.
- Missing values are deliberately left as NaN throughout — no imputation. TabFM handles them, and
  `n_na` makes the missingness pattern itself available as a signal.
- `python final/run.py` copies `final/pipeline.py` over the workspace `pipeline.py` (they are already
  identical) and calls the same `judge.load_task` / `judge.run_pipeline` used during the run, at
  `judge.N_EST_FINAL` estimators rather than the 8 used for CV. Expect the run to take much longer
  than a `val` fold, and expect a GPU to be needed for the TabFM pass.