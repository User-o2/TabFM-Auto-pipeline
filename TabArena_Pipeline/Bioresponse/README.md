# Bioresponse — final pipeline

## What it does

The task is 1776 numeric molecular descriptors, 2500 training rows, scored as `1 - AUC`. The judge
accepts at most 500 columns into TabFM's context, so the whole job is choosing what the frozen
model gets to see. `engineer` ranks the descriptors by ExtraTrees impurity importance averaged over
two seeds and walks that ranking greedily, skipping any candidate whose absolute correlation with an
already-picked column exceeds 0.9, until 220 raw descriptors are kept. To keep the ~1500 discarded
columns from vanishing entirely, it standardises the full matrix over train+test features (no
labels) and appends 40 TruncatedSVD components of it — 260 columns in total. TabFM then predicts by
in-context learning over those rows, with `n_feature_crosses='sqrt'` (the one ensemble knob that
moved the score). A RandomForest(entropy, 500 trees) fitted on all 1776 raw descriptors is trained
alongside but deliberately kept *out* of the context — offered as a column it hurts badly, because
the cross-fitted train values do not match the full-fit test values — and instead its ranking is
mixed into TabFM's ranking in `postprocess` with weight 0.05. `folds` is 5-fold stratified rather
than the judge's default 3-fold, so each fold's 2000 context rows sit closer to the 2500 the final
run fits. Validation numbers for this and for the couple of dozen variants that were tried and
rejected are tabulated in the docstring at the top of `pipeline.py`.

## What build.sh builds

Nothing. `final/build.sh` is a comment and `exit 0`: the pipeline recomputes every derived artefact
(importances, selection, SVD basis, auxiliary forest) inside `engineer` on each run, and the only
files it names — `.side_rf.npy` and `oof_dump.csv` — it writes itself. It takes no time and no disk.

    bash final/build.sh && python final/run.py

## Things a reader could get wrong

- **`.side_rf.npy` is intra-run state, not a cache.** `engineer` writes it, `postprocess` reads and
  deletes it in the same pass. Do not run two passes concurrently in this directory, and do not
  treat a leftover file as reusable — it would be silently ignored anyway if its length does not
  match the current test block.
- **`oof_dump.csv` is append-only and already exists here** from the search. Nothing downstream
  reads it; it was written so blend weights and paired-bootstrap standard errors could be measured
  offline against the training labels without spending evaluations. New runs append to it. Delete it
  if you want a clean record.
- **The blend weight is small on purpose.** `RF_WEIGHT = 0.05` is worth about 0.0001 here. Its
  offline optimum was 0.15 at the 1667 context rows of a 3-fold split and 0.05 at 2000; since the
  final fits 2500 rows and the RF's useful share shrinks as TabFM's context grows, 0.05 is the
  conservative end of a flat curve, not a tuned peak.
- **Most of the variant table is one plateau.** A paired bootstrap over the dumps puts the standard
  error of a difference at roughly 0.0007–0.0013, i.e. larger than most of the gaps in the docstring.
  Only the crosses and the small RF blend clearly survived, together worth about 0.001. Reading
  individual entries there as ordered rankings would overfit the record.
- **`engineer` has a degenerate branch.** When it sees a single class (only `judge.py check`'s
  150-row slice does) it skips the importance model and the forest and selects on a zero importance
  vector. That path exists to keep the free dry run working; it is not a prediction path.