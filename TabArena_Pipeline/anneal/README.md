# anneal — final pipeline

## What it does

The task is 598 training rows × 38 mostly-categorical columns, 5 classes, scored by log loss. TabFM
does all the predicting; everything that pays sits in how the table is handed to it. `engineer()`
rebuilds the frame from scratch: every categorical level becomes its own one-hot column, the 14 flag
columns become indicators, the 7 single-valued columns are dropped, and four more columns that carry
only 1–2 positive rows in the whole dataset go too. Five of the seven numerics use 0 to mean "not
applicable" rather than as a magnitude, so those zeros become NaN (hardness is the exception and keeps
its zeros); thick/width/len are rounded to the step that merges imperial-to-metric measurement jitter,
so physically identical products collapse to one context entry. Three one-hot groups whose missing
level is a class in its own right — surface-quality, condition, steel — get an explicit `=NA`
indicator, because "not applicable" is otherwise the all-zeros pattern and costs the model an AND over
the whole group. One engineered column, `min_freq`, is added: for each row, the count of training rows
sharing the rarest raw value it carries, clipped to a cap and log1p'd — entirely label-free, and the
largest single gain after the encoding, since the dataset's few label exceptions all carry a
value seen once. All of this is fitted on the training fold only (levels, frequency tables, cap).
TabFM then runs with plain default constructor knobs — every one was measured and none beat the
default — and `postprocess()` applies a single global exponent of 0.87 to the output probabilities,
which is cheap insurance at this dataset's ~0.35% rate of confidently-wrong label exceptions.

`folds()` overrides the judge's 3-fold CV with repeated stratified 10-fold × 3 repeats, because at 598
rows the 3-fold spread (0.029 around a mean of 0.028) could not resolve any change. Reported CV under
those folds: 0.01168 log loss at 99.67% accuracy, against 0.02151 for an identity pipeline on the same
folds.

## What build.sh builds

Nothing. `pipeline.py` opens no file — all its input arrives as dataframe arguments, and every fitted
quantity is derived inside `engineer()` from the training fold it is given. `final/build.sh` says so
and exits 0. It takes no time and no disk.

Run:

    cd <this workspace> && bash final/build.sh && python final/run.py

Requires the competition data under `data/` (the same layout the run used) and a working TabFM/GPU
environment; `python final/run.py --check` verifies everything short of the TabFM pass.

## Things a reader could get wrong

- **`final/build/` is not build tooling.** Those four scripts are the offline analysis harness from the
  search: `oof.py` and `cmp.py` reload the per-fold probability dumps and rescore/recalibrate them
  without touching a GPU, `aux.py` refits auxiliary models on the identical folds for comparison,
  `setcfg.py` rewrote `TABFM_KWARGS`/`TAG` between evaluations. They are kept as a record; running them
  is not part of reproducing the submission.
- **`postprocess()` writes to `dumps/final/`** (raw pre-exponent probabilities plus a row_id→label map,
  one pickle per call). That is instrumentation for the offline replay above, not an output anyone
  needs; it is written, never read back, and can be deleted. On the final fit it is a handful of small
  files.
- **The exponent is applied after the dump, not before.** The pickles hold TabFM's raw output, which is
  why `cmp.py` can refit POWER offline. Do not double-apply it when reading them.
- **`POWER` and `softmax_temperature` are the same knob**, not two: with `average_logits=True` TabFM
  returns `softmax(mean_logits / T)`. Only `POWER` is used here.
- **The rarity cap scales with fold size** (`RARITY_CAP_FRAC = 15/538`), so a 538-row CV fold and the
  598-row final fit sit at 15 and 16 rather than at different points of the feature's curve. Replacing
  it with an absolute constant silently changes the configuration.
- `pd.set_option("mode.string_storage", "python")` at import is load-bearing: the judge process cannot
  unpickle a pyarrow-backed string column.