# jm1 — TabFM pipeline

Task: NASA MDP software defect prediction (TabArena `jm1`), ~5.8k software modules described by
21 McCabe/Halstead complexity metrics, binary target `defects`, scored as `1 - ROC-AUC` (lower is
better) averaged over the 9 folds the published TabFM baseline used.

## What the pipeline does

The prediction is a single frozen TabFM in-context pass over the *entire, unmodified* training
table — the search found that every intervention on the rows made things worse (shrinking the
context, collapsing the 415 exact-duplicate metric groups, dropping hard rows, oversampling the
19% positives, bagging subsamples), and so did every label-derived feature (a GBDT out-of-fold
prediction, a leave-one-out group defect rate) and every rank-blend of a side model into the
output. So `preprocess` and `postprocess` are identity, `sample` is not defined at all, and the
only TabFM knob set is `class_shift=False` (with two classes the per-member label permutation
buys no diversity).

All of the gain is in the **feature representation**, done in `engineer`:

1. **Drop `l`, `i`, `b`, `t`.** They are exact algebraic functions of `v`, `d` and `e`
   (`l=1/d`, `i=v/d`, `b=v/3000`, `t=e/18`; mismatch rate under 0.03% each), so they add no
   information and cost a fifth of the table's width.
2. **Keep the 13 raw measurements and add 22 derived columns.** `loc`, `v(g)` and the Halstead
   terms are all proxies for size; what distinguishes a buggy module from a merely large one is
   *shape* — cyclomatic density, comment/blank/code density, operator repetition, vocabulary
   reuse — plus the residuals of the identities that ought to hold
   (`branchCount = 2·v(g) − 1`, `v = N·log2(η)`, `e = v·d`, `loc = lOCode+lOComment+lOBlank`).
   A group ablation showed all five groups contribute.
3. **Signed-`log1p` the derived columns.** They span as many decades as the raw ones
   (`rep_op`, `e_per_loc`), and outlier clipping leaves no resolution inside the bulk. Applying
   the reshape to the raw columns instead is worse — it has to land on the derived quantities.

`folds()` is plain 5-fold stratified CV (seed 2): rows are independent modules with no time or
group structure, so that mirrors how the held-out split was made.

Validation history (3-fold CV, 8 estimators, `1-auc`): identity pipeline ≈ 0.2219, dropping the
redundant columns 0.2199, adding the ratios/residuals 0.2185, log-shaping them 0.2167. The
differences here are the size of the fold noise, so the last step was re-checked on three
independent fold partitions (5-fold seed 0, 10-fold seed 1, 5-fold seed 2) rather than one; it
came out −0.0018 / +0.0002 / −0.0009, a real but small edge. End to end the pipeline is worth
roughly 0.005 AUC over the identity baseline.

## What build.sh builds

Nothing. `final/pipeline.py` reads no files — every feature is computed arithmetically from the
columns the judge already passes in. `final/build.sh` therefore documents that and exits 0; it
takes no time and needs no disk. Run:

```
bash final/build.sh && python final/run.py
```

`python final/run.py --check` verifies prerequisites and imports without touching the GPU.

## Things a reader could get wrong

- **`final/build/sweep.py` is not a build step.** It is the search driver: it rewrote the `CFG =`
  line of the *then-current* `pipeline.py` for each variant fed on stdin, ran `judge.py check`
  and `judge.py val`, and left the best variant in place. The scored pipeline has no `CFG` line,
  so running sweep.py against it would fail its assertion. It is shipped as a record only.
- **Earlier snapshots wrote a `.blend_*.npy` cache**; the scored pipeline (`history/pipeline_096.py`)
  removed that side-model path entirely, so no stale cache is read and none needs rebuilding.
- **`_shape()` reads `n`, `v`, `d`, `e` from the input** even though they are not in `BASE` — they
  are used to build the derived columns and then dropped from the output table. `X['d']` and the
  rest must therefore still be present in the raw frame; the pipeline does not tolerate a
  pre-pruned input.
- **`eta.replace(0, np.nan)` in `v_resid` is a pandas call**, so `engineer` expects pandas
  Series, not numpy arrays — which is what the judge passes.
- The run's reported score comes from the mean over the 9 TabArena folds of the final evaluation,
  not from the 3-fold CV numbers quoted above; the two are not directly comparable.