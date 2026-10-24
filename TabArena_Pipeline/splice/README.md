# splice — donor / acceptor / neither, log loss

## What the pipeline does

The task is a 60-nucleotide window centred on a candidate splice junction, classified as `EI`
(donor), `IE` (acceptor) or `N`. TabFM is the predictor throughout; the pipeline only decides what it
sees and how its output is finished.

The 60 raw position columns are kept as low-cardinality **categorical** strings — ambiguity codes
(N/D/S/R) are collapsed to a single `N` category rather than imputed, because the code is itself a
signal — and TabFM is left to encode them. Integer codes for the bases score much worse. On top of
those columns, `engineer` adds ~160 hand-built numeric and categorical summaries that TabFM cannot
form by attending to one column at a time: the two junction dinucleotides and their GT/AG flags, two
longer consensus windows around the junction, AG/GT counts and distances to the nearest flanking
site, per-frame stop-codon counts and per-phase GC (the coding-potential signature that separates
exon from intron), 48 dinucleotide counts, base composition in five-base blocks, polypyrimidine-tract
maxima and homopolymer runs. All 2126 training rows go into the context as one draw — subsampling,
deduplicating, augmenting and reweighting rows were all tried and all cost accuracy.

Alongside TabFM, `engineer` fits a small **order-2 positional weight-array model** (a per-class,
per-position Markov model of the sequence, order-2 backed off to order-1 backed off to the position
frequency, averaged over three backoff strengths). It is far weaker alone (0.1295 vs TabFM's 0.0806)
but wrong in different places, so `postprocess` combines the two geometrically with weight 0.16 on
the WAM and then mixes in a 0.002 uniform floor. Log loss here is decided by a few dozen confidently
wrong rows, and both steps target that tail: 0.0806 raw → 0.0794 with the floor → **0.0767** blended,
on stratified 10-fold. TabFM's own ensemble knobs contribute the rest: `permute_categorical`, five
different normalisations across the eight members, and NNLS-fitted out-of-fold member weights.

## What build.sh builds

Nothing. The pipeline has no prerequisite caches — it reads only `.wam_cache.npz`, which `engineer`
writes and `postprocess` reads back within the same fold. `build.sh` says so and exits 0; it takes no
time and no disk. Running it before `python final/run.py` is harmless and unnecessary.

## Things a reader could get wrong

- **`oof_dump.pkl` is not an input.** `postprocess` appends raw per-fold TabFM probabilities to it for
  offline analysis. It is append-only and never read by the pipeline; deleting it changes nothing. It
  does grow across repeated runs.
- **`final/build/` is search tooling, not build tooling.** `try.py` edited `TABFM_KWARGS` and ran
  check+val; `lab.py` replays the judge's 10-fold log loss over dumped TabFM probabilities and is how
  `BLEND_W` and `SMOOTH` were fitted. `lab.py` expects `$TMPDIR/oof_P.npy`, a file distilled from
  `oof_dump.pkl` by an ad-hoc shell snippet during the run; it is not needed to reproduce the
  submission, and the constants it produced are already in `pipeline.py`.
- **`folds()` is 10-fold, not the judge's default 3.** Every score quoted above and in the
  `pipeline.py` docstring is on that stratified 10-fold split with 8 estimators, so it is not directly
  comparable to a 3-fold number from elsewhere in the run.
- **String dtypes matter.** `_obj()` forces plain numpy-object strings on every frame the hooks
  return, because the judge process that unpickles the hook output has no pyarrow and cannot load a
  pandas 3 Arrow-backed string column. Removing that cast breaks the run in a way unrelated to
  modelling.
- Most of the constants near the top of `pipeline.py` (`IMPUTE_N`, `SPLIT_N`, `DEDUP`, `DIMER_COLS`,
  `POS_LLR`, `TOPK`, `EMPHASIS`, `NUMERIC_POS`) are switched **off** and kept only as documented
  negative results, with the measured cost of turning each one on in the adjacent comment.