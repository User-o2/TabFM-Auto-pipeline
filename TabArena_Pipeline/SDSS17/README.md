# SDSS17 — ObjectType (GALAXY / QSO / STAR), log loss, lower is better

## What the pipeline does

The prediction is made entirely by frozen zero-shot TabFM; the pipeline only changes *what it sees*.
Two things happen. First, the 11 raw columns are rewritten into an astronomically natural basis: the
two positional columns that carry no class signal but do carry survey-footprint noise (`cam_col`,
`fiber_ID`) are dropped, and the five photometric magnitudes `u g r i z` are replaced by four colours
(`u-r`, `g-r`, `r-i`, `r-z`) plus one brightness anchor, `mean(g, r, i)`. Colours are what separate
quasars from stars and galaxies — a magnitude difference is distance-independent, whereas the raw
magnitudes are five strongly collinear columns that mostly encode apparent brightness. Anchoring on
the mean of the three best-measured SDSS bands rather than on a single band keeps the same brightness
information with less photometric noise. Everything else (`alpha`, `delta`, `redshift`, `plate`,
`MJD`, `spec_obj_ID`) passes through unchanged, so TabFM gets 9 columns instead of 11. Second, the
prediction is bagged over the training context: `sample()` returns five index sets — one full draw
plus four independent 95 % subsamples (seeded, `default_rng(0)`) — and TabFM runs once per draw with
`average_logits=True`, `softmax_temperature=1.0` and quantile normalisation, so the five in-context
passes are averaged in logit space. No post-processing is applied to the probabilities.

Validation log loss on the harness's 3-fold CV was **0.05755**, against published TabFM baselines of
0.069689 (plain) and 0.068171 (ensemble) on the same model and folds.

## What build.sh builds

Nothing. `final/build.sh` is a no-op that exits 0. The pipeline reads no caches, no auxiliary
artefacts, and no files at all — it is four pure hooks over the DataFrames the judge passes in, using
only numpy. It needs no disk beyond the raw `data/train.csv` and `data/test.csv` and takes no time.

```
cd <this workspace> && bash final/build.sh && python final/run.py
```

## Things a reader could get wrong

- **The cost is all TabFM, and it is 5×.** `sample()` returning five index sets means five full
  in-context passes per estimator, over ~52k and ~49k context rows. The hooks themselves are
  microseconds; the wall clock is entirely the foundation-model passes, so budget accordingly.
- **`DROP` is applied before the colour arithmetic** and only names `cam_col` and `fiber_ID`; the
  judge's `row_id` is added to X and removed again by the judge itself, so the pipeline never sees or
  drops it. If you run these hooks outside `judge.py`, do not pass `row_id` through to `engineer`.
- **The raw magnitudes do not survive `engineer`.** `u g r i z` are dropped after the colours and the
  anchor are computed; TabFM never sees an individual band.
- **The subsample seed is fixed at 0 and the first draw is always the full training set**, so results
  are reproducible run to run, but the bagging is *not* an out-of-bag scheme — the draws overlap
  heavily by design (95 % each). They exist to decorrelate TabFM's in-context attention, not to
  estimate variance.
- **`FRAC = 0.95` was tuned against 3-fold CV on this task's fold-0 training split.** It is a
  small-effect knob (differences of ~0.0005 in log loss); do not read it as a general recipe.