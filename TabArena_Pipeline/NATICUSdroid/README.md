# NATICUSdroid — final pipeline

## What it does

The task is 4994 Android apps described by 86 binary permission flags, predicting `Malware`
(Yes/No), scored as `1 - AUC`. TabFM z-scores every column before attending over the context rows,
so for a binary flag an affine re-encoding is a no-op: the only two things about a flag that reach
the model are *which of its two states is the minority* (that state gets the large |z|) and *whether
the flag is routed as numeric or as categorical*. The pipeline sets exactly those two things, per
flag, and changes nothing else — no columns are added, removed or transformed, and the context is
always the full set of training rows.

Inside each training fold it measures every flag's signed lift `P(malware | flag=1) - P(malware |
flag=0)`. The top 80% by |lift| are kept **numeric and sign-aligned**: any flag whose lift is
negative is replaced by `1 - flag`, so a large positive value always means the malware-associated
state of that permission. The remaining 20%, whose sign is not reliable enough to fix without
fitting noise, are cast to `category`, which turns on TabFM's `permute_categorical`: each ensemble
member then sees its own random polarity for them and the ensemble averages over a symmetry the data
has and the model does not. On top of that, `n_feature_crosses='sqrt'` gives half the members nine
random pairwise crosses — built from numeric columns only, so they cross the *oriented* strong
flags, which is why the same knob is a loss on the raw table. `folds()` is 2 repeats of a stratified
10-fold (seed 3), giving 4495 context rows per fold, the closest the training set gets to the 4994
the final fit sees.

Measured against the identity pipeline on the same folds, at the geometry closest to the final's
(4495 context rows, 20 folds, a seed that took no part in any design choice): **0.011552 vs
0.012029**, better on 18 of 20 folds, t = 4.6. The full ablation and eval log is in the docstring at
the top of `pipeline.py`.

## What build.sh builds

Nothing. `pipeline.py` reads no files — every quantity it uses is recomputed from the training
DataFrame it is handed. `final/build.sh` is a comment and `exit 0`; it takes no time and no disk.

    bash final/build.sh && python final/run.py

## Things a reader could get wrong

- **`data/train.csv` is sorted by label** (only 3 label switches across 4994 rows), so row position
  predicts the target almost perfectly *inside the training file*. That is an artefact of how the
  file was written — `test.csv` is shuffled (position/prediction Spearman −0.002) — and any
  index-derived feature would show a large fake CV gain. The pipeline deliberately derives nothing
  from row order.
- A consequence of that sorting: `judge.py check`, which runs the hooks on the first 200 rows, sees
  a single class. `engineer()` detects this and returns the table untouched, so `check` exercises
  the plumbing but not the logic. `final/build/t.py` is the local substitute — it runs the hooks on
  a real two-class fold and pushes the result through TabFM's own encoder. Run it from the
  workspace root, not from `final/`.
- The lift ranking is computed **inside each fold**, from that fold's training rows only, so there
  is no leakage from validation rows and the shipped code path is identical in CV and in the final
  fit (only the number of context rows changes).
- `STRONG_FRAC = 0.8` is the single hand-set constant. It is flat between 0.75 and 0.9 at 3995
  context rows and a clean interior optimum at 4495 (0.65 → 0.011719, 0.8 → 0.011552, 0.9 →
  0.011709); its advantage grows with context size, and the final has more rows than any tuning run,
  which is why 0.8 ships.
- TabFM stays the only predictor. No auxiliary model's output enters the features or the prediction
  (a GBM+ExtraTrees rank blend was tried and cost 4e-4).