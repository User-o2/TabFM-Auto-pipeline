# COIL2000 / MobileHomePolicy — final pipeline

Metric: 1 − ROC-AUC, lower is better. Final validation score 0.21445 (12-fold), against 0.22008 for
the identity pipeline.

## What the pipeline does

TabFM predicts, unmodified: `TABFM_KWARGS` is empty, because all sixteen constructor knobs were
measured one at a time during the run and none of them beat the shipped defaults. Everything the
pipeline does is to the table TabFM reads.

Three columns arrive as decoded strings (`avgAge`, `romanCatholic`,
`contributionPrivateThirdPartyInsurance`) even though their sibling columns carry the same ordinal
scales as plain integers; `preprocess` maps them back onto that numeric scale, so a ladder is
contiguous rather than half-categorical. `engineer` then appends seventeen row-wise **count**
columns and drops nothing else. The table already states, per product, how much a household
contributes and how many policies it holds, and per zip-code ladder what share of the neighbourhood
sits in each band — but never *how many* products the household holds, or across how many bands a
ladder spreads. Those cross-column counts are the one summary an attention-over-rows model cannot
form from a single row, and they are essentially the entire gain (removing them costs +0.006). The
counts are: distinct products held, total policies, policies per product, how many of the 36
percentage columns are zero and how many are ≥5, how many income and education bands are occupied,
and the band spread of each of the ten socio-economic ladders. Finally, for each product held by
fewer than 5% of households the `numberOf*` half of the pair is dropped — it correlates 0.86–0.98
with the contribution half and adds nothing for rare products.

`folds()` overrides the judge's default 3-fold CV with `RepeatedStratifiedKFold(4 × 3)`. At a 6%
positive rate the per-fold std is ~0.010, so 12 folds are what make the paired per-fold comparison
sharp enough (SE ≈ 0.0004) to separate a real effect from a re-roll of TabFM's ensemble. `postprocess`
is the identity — AUC is invariant to any monotone rescaling of the scores.

The docstring at the top of `pipeline.py` is the fuller record: what was tried, what it scored, and
which of the small edits were ablated more than once.

## What build.sh builds

Nothing. `final/build.sh` is a documented no-op that echoes one line and exits 0.

The pipeline names no file at all — its hooks operate only on the DataFrames the judge hands them,
and the judge loads those from `data/train.csv` and `data/test.csv` itself. There are no caches, no
precomputed features, no auxiliary models, and the run wrote no helper scripts (every experiment was
an edit to `pipeline.py` followed by `python judge.py val`), which is why `final/build/` is empty.
Runtime: instant. Disk: zero.

## Things a reader could get wrong

- **The reproduction is still not bit-identical.** `run.py` runs one frozen TabFM pass with
  `judge.estimators_final` estimators; validation used 8. Ensemble member ordering re-rolls, and this
  task's re-roll floor is ~0.002 in the metric — bigger than several of the deliberate design
  decisions recorded in the docstring. Do not read a small delta against 0.21445 as a bug.
- **Any edit that changes the emitted column count re-rolls that ensemble.** This is measured, not
  suspected: eval 22 wrote the same two normalisers in the other order — statistically the identical
  ensemble — and moved the score 0.0014 with 10 of 12 folds agreeing in sign. It is the resolution
  limit on everything below ~0.002.
- **`engineer` assigns columns one at a time on purpose.** `pd.concat` rebuilds the frame and
  destroys the object-dtype string columns that `_decode` deliberately preserves; likewise `_decode`
  casts the two nominal columns to `object` because arrow-backed strings do not survive the pickle
  the judge uses to cross the hook boundary. Neither line is stylistic.
- **The 0.05 rare-product cut sits just under life insurance (5.2%).** A CV fold whose training rows
  fall slightly short of that drops a seventeenth column, so the cut is mildly fold-dependent. Moving
  it to 0.04 — identical on the full training set — was measured (0.21462 vs 0.21451) and rejected;
  both are the same model on the full data.
- **`EXTRA_COUNTS` and the band-midpoint rescaling look like unfinished work; they are not.** Both
  were implemented, measured repeatedly and removed. The apparent gain from an 18th count in evals
  78–80 was an artifact of `DROP_RARE_COUNTS` having been left at 0 from the preceding ablation.
- **`run.py` refuses to run if `<workspace>/pipeline.py` differs from `final/pipeline.py`.** They are
  identical as shipped; if you edit one, reconcile both.