# website_phishing — final pipeline

## What it does

`website_phishing` is a tiny table: 902 training rows, 451 test rows, nine categorical indicators
that each take one of three levels (`Phishy` / `Suspicious` / `Legitimate`), and a three-class
target scored by multiclass log loss. With that little data the useful lever is not feature
engineering but *representation* plus TabFM's own ensembling.

The pipeline does one thing to the features: it recodes every indicator as an **ordinal number**
(`Phishy = -1`, `Suspicious = 0`, `Legitimate = 1`) instead of leaving it as a string category. The
levels have a natural low/middle/high order — the other two orderings were tried and were worse —
and, more importantly, TabFM's automatic feature crosses only fire on numeric columns, so the
ordinal coding is what lets it build interactions between the nine indicators at all. Everything
else is TabFM configuration: per-member logits (not averaged probabilities) so that a downstream
combiner can see them, out-of-fold **vector calibration** of the multiclass logits with no ridge
penalty on the calibration parameters, **NNLS** ensemble-member weights shrunk most of the way back
to uniform (`nnls_beta = 0.25`), and a slightly cool softmax (`temperature = 0.85`). `preprocess`
and `postprocess` are identity.

`folds()` is overridden to 3×10-fold repeated stratified CV (seed 1). Ten folds means each fit sees
811 of the 902 rows, close to what the final gets, and averaging 30 folds is quiet enough to read
differences around 0.001 — a 3-fold default on 902 rows is not.

One thing deliberately *not* in the pipeline: an earlier revision added a fixed outer calibration
map in `postprocess` and scored 0.21212, but that gain was an artifact. Both CV splits used to
"validate" it score the same 902 rows, so the check was in-sample; the only row-disjoint test
available (fit the map on half the rows, score the other half) cost +0.007, and +0.010 for a full
3×3 matrix map. Six to twelve parameters read off 902 rows is sampling noise. It was dropped, and
the fold seed was moved so the final would be selected over evaluations that did not carry it.

Scored result on the held-out split: **log loss 0.2067** (30-fold TabArena mean 0.2062, against
published TabFM baselines of 0.2108 plain / 0.2091 ensemble).

## What build.sh builds

Nothing. `final/build.sh` is a no-op that exits 0, and it says so in a comment.

The pipeline reads no files — no cached features, no fitted artifacts, no auxiliary tables. It
consumes only the DataFrames the judge passes to its hooks. The 26 scripts under `final/build/`
(`v1.py` … `v22.py`, `lab.py`, `bench.py`, `post.py`, `calib.py`) are the offline experiments that
chose the constants now hard-coded in `TABFM_KWARGS` and `ORD`; they wrote only their own scratch
output (`lab_results.jsonl` and similar), which nothing in the pipeline reads. They are kept for the
record, not as build steps.

Runtime: `bash final/build.sh` is instantaneous and needs no disk. `python final/run.py` is the
whole cost — roughly a couple of minutes on one GPU for 8 TabFM estimators over a 902-row context —
and writes a single `final/submission.csv`.

## Things a reader could get wrong

- **`ORD` reuses the target's class names as level names.** The three indicator levels are literally
  spelled `Phishy`, `Suspicious`, `Legitimate`, the same words as the three target classes. They are
  unrelated to the label; the mapping is a feature encoding, not leakage.
- **The `row_id` column is skipped** by `engineer`, on purpose — it is the submission key, not a
  feature.
- **`average_logits: False` is load-bearing.** Both the vector calibrator and the NNLS member
  weighting need per-member logits; flipping it back to `True` silently disables the two settings
  underneath it.
- **`folds()` affects selection, not the final fit.** It defines the CV the search was scored on; the
  final fit uses all 902 training rows.
- The reported 0.2067 is the TabArena 30-fold protocol's held-out score, not a Kaggle leaderboard
  number — `data/meta.json` lists no medal thresholds for this task.