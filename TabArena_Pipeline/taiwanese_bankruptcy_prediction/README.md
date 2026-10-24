# Taiwanese bankruptcy prediction — final pipeline

## What it does

TabFM makes the prediction; the pipeline is about what it gets to read. `engineer` appends 14
combinations of the raw financial ratios to the 94 originals — mostly *gaps* between pairs of
near-duplicate readings (the three ROA variants, the two gross margins, the two net-value-per-share
columns, pre- vs after-tax interest), because those pairs correlate above 0.999 and a per-column
rank transform provably cannot reconstruct the difference between them. A random forest fitted on
the fold's own training rows (gini importance averaged over three seeds, `min_samples_leaf=3`)
ranks all 108 columns and the top 52 are kept. Those 52 are mapped to gaussian ranks by a single
`QuantileTransformer` fitted on the training and test rows pooled, with one knot per row — pooling
is legal here (`engineer` sees both frames and no labels) and it is what makes the raw columns,
21 of which are bimodal with mass at 1e9, readable at all. TabFM then runs as a 32-member ensemble
over `norm_methods=['none','power','quantile','robust']`. Finally `postprocess` blends an
ExtraTrees classifier — fitted on the same fold's rows but reading all 108 columns, i.e. paid for
being unlike TabFM rather than for being strong — into TabFM's ranking at weight 0.12, in probit
space rather than rank space so the blend is decided in the tails where the 3.2% of bankruptcies
live. The result stays ~0.998-correlated with TabFM's raw ordering.

CV: 0.05329 (1 − AUC, lower is better) over 9 folds, RepeatedStratifiedKFold 3×3, against 0.05806
for the identity pipeline. `folds()` uses 3×3 rather than the judge's default 3-fold because a
plain 3-fold left a 0.0086 standard error on the mean — larger than every effect measured in the
run.

## What build.sh builds

Nothing. The pipeline reads no precomputed cache: the one file it names, `.aux_pred.npy`, is
written by `engineer` and read back by `postprocess` in the next hook subprocess. `build.sh` says
so and exits 0 — it takes no time and no disk. Just run:

```
cd <this workspace> && bash final/build.sh && python final/run.py
```

`run.py` needs the competition data in `data/` and a GPU for TabFM; the full pass is one
32-estimator TabFM run over 3030 context rows, so it is minutes, not hours.

## Things a reader would otherwise get wrong

- **The order of the `assign` calls in `_add_features` is load-bearing**, to about 0.0003. It
  decides ties in the importance ranking and therefore which columns sit either side of the K=52
  cut. Eval 67 is the identical pipeline with the same 14 columns sorted by kind, and it scores
  0.05375. Do not tidy them.
- **`A[:] = qt.transform(...)` assigns in place on purpose.** Rebuilding the frame would give it an
  arrow-backed column Index that the judge's hook subprocess cannot unpickle.
- **`norm_methods` keeping `'none'` is not an oversight.** Once every column arrives as a gaussian
  rank, `'none'` is the member that leaves the good representation alone; TabFM's own default
  `['none','power']` ties it exactly, and the four-method list ships only because it spreads the
  32 members over more views for the same score.
- **The blend weight and the ExtraTrees are secondary.** 0.12 → 0.18 → 0.20 are all ties. The
  geometry (probit, not rank) is what was worth 0.0001–0.0002, not the weight.
- **Class rebalancing hurts here.** The context is 3.2% positive and every attempt to change that
  lost: balanced draws +0.0011, repeating positives +0.0124, dropping hard negatives +0.0015.
  Likewise stacking the tree score in as a *feature* cost 0.0074; only the output blend works.
- `pipeline.py` in the workspace root and `final/pipeline.py` are the same file (`history/
  pipeline_087.py`); `run.py` refuses to proceed if they ever diverge.