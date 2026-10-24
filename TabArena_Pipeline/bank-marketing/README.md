# bank-marketing — final pipeline

## What it does

The predictor is a frozen TabFM doing in-context learning over the whole training fold; the search
that produced this pipeline was almost entirely a search over *how the table is presented to it* and
over its ensemble knobs, not over added modelling machinery. Two things survived.

**Feature side — compression, not addition.** The dataset is 13 raw columns (no `duration`), and
TabFM turned out to be sensitive to column *count* here: eight OOF target-encoded columns cost
+0.0034 in metric, four log-scaled count columns +0.0005, while dropping `default` outright was
exactly free. So the only engineering step is a merge: the three yes/no binaries `housing`, `loan`
and `default` are concatenated into one 8-level categorical `hld`, carrying the same information in
one column instead of three (13 → 11 columns). Nothing is scaled, imputed, encoded or sampled —
`preprocess` and `postprocess` are identity, and there is no `sample` hook, so every ensemble member
attends over the entire training fold.

**Ensemble side — TABFM_KWARGS.** Four normalisation views (`none`, `power`, `quantile`, `robust`;
dropping `none` costs 0.0011), probability averaging rather than logit averaging, NNLS stacking of
the members (`nnls_beta=1.0`, worth ~0.0006), `n_svd_features="sqrt"` (~0.0002),
`permute_categorical=True` (the single largest win, ~0.0007, better on all three folds), and
`softmax_temperature=1.4` — a deliberate softening of TabFM's output distribution that sits on a
flat plateau between 1.4 and 2.2 and is worth ~0.0004.

Things that were tried and rejected, so a reader does not re-run them: feature crosses, per-member
row bagging and fractional bagging, class-balanced or positive-duplicated contexts, reduced
`max_num_rows`, frequency-mode categorical encoding, pdays sentinel / signed-log balance / log-count
transforms, OOF target encoding, and a HistGB blend on the dumped OOF probabilities (it reached only
0.2455 alone, and the best blend bought +0.0002 for ~900s per fold).

3-fold CV on the training set, metric `1-auc` (lower is better): **0.23607**, down from 0.23920 for
the identity pipeline.

## What build.sh builds

Nothing. `final/build.sh` is a comment and `exit 0`. The pipeline reads no files it does not receive
as function arguments — no caches, no fitted artifacts, no precomputed features — so there is no
build step to reproduce. It takes no time and no disk. `final/build/` is therefore absent: none of
the throwaway scripts written during the search is a prerequisite of the scored pipeline.

## Things a reader could get wrong

- **The `object` dtype in `_feat` is load-bearing, not stylistic.** The engineered `hld` column is
  built with `pd.Series(..., dtype=object).astype(object)` because the hook sandbox has pyarrow and
  the judge server that unpickles the hook output does not: a freshly created pandas-3 `str` column
  is arrow-backed inside the sandbox and cannot be reconstructed on the other side. Two evaluations
  (RESULT 31 and 32) died on exactly that. Do not "clean this up" to a plain string dtype.
- **`MERGE` is applied defensively.** `_feat` only merges when all three source columns are present,
  so the hook is a no-op on a frame that has already been transformed. That is guard code, not a
  sign that the columns are ever legitimately missing in this task.
- **Row order and the fold definition matter for reproducing the CV number, not the submission.**
  The pipeline defines no `folds()`; the 0.23607 above is against the judge's default 3-fold split
  (`folds_sig=d4d7ad64`). `python final/run.py` does not recompute it — it runs the final 8-estimator
  TabFM pass over the full training set and writes `final/submission.csv`.
- **`run.py` copies `final/pipeline.py` over the workspace `pipeline.py`** (the judge's `apply_hooks`
  loads it by that name) and refuses to run if the two differ, rather than silently overwriting the
  scored record.