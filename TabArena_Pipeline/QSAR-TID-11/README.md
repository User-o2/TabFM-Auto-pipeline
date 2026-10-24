# QSAR-TID-11 — TabFM blended with a Tanimoto/Tversky-kernel SVR

The task is regression of `MEDIAN_PXC50` from 1024 binary FCFP6 fingerprint bits, scored by RMSE
(lower is better), on ~3.8k training rows.

The pipeline is a fixed-weight blend of two predictors over the same rows. TabFM is the main one, but
it can only take 500 columns, so `engineer` ranks the 1024 bits by variance and hands it the top 500 —
that is the only thing done to the features; there is no encoding, scaling, or synthetic column.
Alongside it, `engineer` fits a chemistry-appropriate similarity model that the column cap does not
constrain: an SVR (`C=10`, `epsilon=0`) on a precomputed kernel that is the average of the Tanimoto
coefficient and a symmetric Tversky coefficient with `alpha=0.7`, i.e. two ways of trading off shared
bits against bits only one molecule carries. That kernel model is itself a small blend: 0.90 of it is
computed over all 1024 bits, and 0.10 over the 524 low-variance bits *alone* — the block TabFM never
sees. That second term is a weak model on its own but is the most decorrelated component available
(error correlation ~0.69 with TabFM, against ~0.87 for the full-bit kernel), precisely because it is
built from what TabFM is blind to. `engineer` writes the kernel prediction to `.blend.csv` keyed by
`row_id`; `postprocess` reads it back, joins on `row_id`, and returns `0.66 * TabFM + 0.34 * kernel`.
Every weight is a fixed constant, chosen by cross-validation during the search, not refit at predict
time; if the kernel side fails or the join is incomplete, `postprocess` silently falls back to the
raw TabFM prediction.

Scored result: mean RMSE **0.7271** over the 9 TabArena folds, against the published TabFM baselines
of 0.7548 (plain) and 0.7329 (ensemble). Search-time 3-fold CV was 0.7370.

## build.sh

There is nothing to build. `build.sh` is a comment and `exit 0`. The pipeline reads no cache, no
precomputed feature file, and no fitted artifact — only the frames the judge passes to its hooks. The
one filename it mentions, `.blend.csv`, it creates itself inside `engineer` and consumes inside
`postprocess` on the same fold. Zero time, zero disk.

To reproduce:

    cd <this workspace> && bash final/build.sh && python final/run.py

## Things a reader could get wrong

- **`.blend.csv` is state, not a cache.** It is the channel between `engineer` and `postprocess`
  within one fold, rewritten every time `engineer` runs. It must not be pre-seeded, and a stale copy
  left in the workspace is harmless only because the join is on `row_id` and is checked for
  completeness before use. Do not run two folds concurrently in the same working directory.
- **`preds_dump.csv` is append-only diagnostic output** from the run's search, not an input, not a
  submission. Nothing reads it. It will keep growing across runs; deleting it changes nothing.
- **The 500-bit cap is TabFM's, not a modelling choice.** The variance ranking is a way to spend the
  budget, and the run measured that alternatives (an LSA summary of the discarded block, selecting
  bits by |corr| with the target) were both worse. The discarded bits are recovered on the SVR side
  instead — that is what the `W_RARE = 0.10` term exists for.
- **The weights are dataset-specific constants.** `W_TABFM = 0.66`, `W_RARE = 0.10`, `TVERSKY = 0.7`,
  `SVR_C = 10` were all tuned on this training set's CV folds; none of them adapt.
- **Cost scales quadratically.** The kernel is dense over all training rows (`n x n` Gram plus
  `n_test x n` for prediction), computed twice per fold — once on 1024 bits, once on 524. At ~3.8k
  training rows that is negligible; on a much larger split it would dominate, and it would also hit
  the judge's 600 s per-stage limit long before TabFM did.