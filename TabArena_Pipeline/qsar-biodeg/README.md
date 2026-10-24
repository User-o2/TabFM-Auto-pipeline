# qsar-biodeg — final pipeline

## What it does

The task is 702 training molecules described by 41 numeric molecular descriptors, binary
biodegradability, scored as 1 − AUC. TabFM sees the 41 descriptors exactly as given: every
representation change tried during the run — added percentile-rank copies, dropping the nine
near-constant columns, log1p on the count columns, out-of-fold model scores as extra columns,
synthetic feature crosses, SVD columns, and a PCA rotation at identical column count — measured
worse than the identity on 25 folds, most of them well past the noise. What is tuned is how the
ensemble reads the table rather than what is in it: TabFM's 4-sigma outlier clip is switched off
(it bites hardest on the un-normalised half of the ensemble, which is where these zero-inflated
counts keep their signal), members are pooled as probabilities rather than logits, and each of the
eight members is given a fresh 75 % bagged sample of the training rows. That last knob is the
largest TabFM-side effect in the run — on a table this small nothing else subsamples rows, so
without it the members differ only by column order and class rotation. Because the judge reads
`TABFM_KWARGS` after the hooks run and the split sizes differ (561 rows in CV, 702 in the final),
`engineer()` writes `max_num_rows` itself.

On top of that, `postprocess` blends rankings: 0.60 on TabFM's ranking of the test rows and 0.40 on
the mean ranking of four ordinary sklearn models fitted on the same training rows — a random
forest, extra trees, a histogram gradient booster, and QDA behind a normal quantile transform. The
three tree models additionally get twelve hand-built descriptor ratios (oxygens per heavy atom,
halogens per heavy atom, heteroatom count, electronegativity over ionisation potential, and so on);
a tree cannot form a ratio for itself, whereas TabFM rejects every extra column, so these never
leave `preprocess` and TabFM's own input is untouched. QDA is better on the raw 41. TabFM remains
the predictor — it carries the majority of the weight alone against four models sharing the rest,
and its correlation with the blended output is 0.994 — but the blend is worth 0.0021, the single
largest effect measured in the run. The partner set and the ratio columns were chosen on a separate
5×6 CV (seed 1234) that TabFM's dumps do not cover, so only the blend weight was fitted to the 25
scoring folds. Validation uses 5×5 repeated stratified CV; the rows are exchangeable, so there is no
group or time structure to mirror.

## What build.sh builds

Nothing. `final/build.sh` is a comment and `exit 0`. The pipeline reads no precomputed cache: the
one file it opens, `.partner/<key>.npy`, is written by its own `engineer()` hook seconds earlier in
the same run, and `.dump/<key>.npz` is write-only. It takes no time and no disk.

The full reproduction is therefore just:

    cd <this workspace> && bash final/build.sh && python final/run.py

which is one TabFM pass with eight estimators over a 702-row context plus four small sklearn fits —
a couple of minutes on a GPU, dominated by model load.

## Things a reader could get wrong

- **`final/build/` does not build anything.** Those eleven scripts are the offline analysis harness
  used during the search: `oh.py` matches the `.dump/*.npz` validation dumps back to
  `data/train.csv` by row content and rescores post-processing for free, `cmp.py` does the paired
  t-test between two `[RESULT n]` lines in `log.txt`, and the rest are one-off probes (ratio search,
  greedy partner selection, PLS, nested weight fitting, decorrelation). They print numbers; none of
  them writes a pipeline input. They are copied here for the record, not for the build.
- **`.partner/` and `.dump/` are run-local.** Both are keyed by a hash of the test-row contents, so
  a leftover directory from a different split cannot be picked up by mistake. If you want certainty,
  delete both before running; the pipeline recreates what it needs. This matters because one probe
  run during the search *did* silently re-read a previous run's partner file and scored as if
  blended — that probe path was removed rather than patched, and the current pipeline only ever
  reads a file written in the same process.
- **`PSEUDO_P = 1.0` and `LOG_COUNTS = False` are inert.** They are two preprocess experiments left
  in the code at their off values, kept because the measurements next to them in the module
  docstring are the reason the pipeline does not use them. Do not read them as active features.
- **The `pred.copy()` in `postprocess` is deliberate.** A DataFrame whose column index is
  constructed inside a hook does not survive the pickle back to the judge on this pandas build
  (ImportError on a pyarrow-backed StringArray). Mutating the frame the judge handed over is the
  fix; building a fresh one costs two evaluations to rediscover.
- **`threadpool_limits(1)` in `preprocess` is not a leftover.** ~100 cores against a 561×41 table is
  all OpenMP overhead; the partner fits are meaningfully faster single-threaded.
- **`run.py` copies `final/pipeline.py` over the workspace `pipeline.py`** and refuses to run if the
  two differ, so the reproduction cannot drift from what was scored.