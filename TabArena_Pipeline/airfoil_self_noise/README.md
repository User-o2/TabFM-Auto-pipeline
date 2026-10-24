# airfoil_self_noise — TabFM pipeline

## What it does

The prediction is made by TabFM, frozen and zero-shot, attending over the training rows; the pipeline
around it is entirely a change of coordinates. The NASA airfoil self-noise data is 106 (chord,
free-stream velocity, angle of attack) configurations, each swept over 1/3-octave frequency bands,
and the physics that sets the sound pressure level is multiplicative and lives on log scales. So
`engineer()` rewrites each row in the variables the Brooks–Pope–Marcolini semi-empirical model is
actually written in: log frequency, log displacement thickness and log chord; the two Strouhal
numbers `f·δ/U` and `f·c/U`; the chord- and thickness-based Reynolds numbers; the thickness-to-chord
ratio; and a `10·log10(δ) + 50·log10(M)` amplitude term that carries the model's overall level
scaling. The raw columns ride along, except `frequency`, `free-stream-velocity` and `chord-length`,
which are dropped — the log and dimensionless forms of those three say the same thing more usefully,
and keeping both only dilutes the attention. The target is left untouched, the full training split is
handed to TabFM as a single draw, `TABFM_KWARGS` is empty (judge defaults), and `postprocess()` is a
pass-through. This coordinate change alone is what the run bought: RMSE 1.377 on the identity
pipeline down to about 1.247.

The module also carries switched-off machinery from the search — rank-Gauss target transform, bagged
or bootstrapped draws, permutation draws, per-configuration leave-one-out neighbour brackets,
configuration mean-encoding, column duplication, prediction clipping. None of it is reachable with
the `OPT` dict as scored; the docstring records why each was dropped (notably: leave-one-out spectral
neighbours scored 1.62, far worse — the 1/3-octave sweeps are too coarse to interpolate across).

## What build.sh builds

Nothing. `pipeline.py` names no file and reads nothing but the DataFrames the judge hands its hooks,
and the run wrote no helper scripts or caches — `final/build/` is empty for that reason.
`bash final/build.sh` prints one line and exits 0. Runtime: instant. Disk: zero.

## Things a reader could get wrong

- **The data does not come from `data/`.** This is a `tabarena:` task. `judge.load_task` reads the
  fold cache at `~/.cache/tabarena/airfoil_self_noise/fold_0/{train,test}.csv` (override with
  `TABARENA_DIR`), and asserts with "fold cache missing; run the tabarena downloader" if it is not
  there. The `data/train.csv` and `data/test.csv` in this workspace are a label-free copy the judge
  materialised for the agent to look at; they are not what a rerun loads. Populating that cache is
  the one prerequisite, and it is upstream of this directory — build.sh deliberately does not try to
  fetch it.
- **The scored number is a 30-fold mean, not this workspace's fold 0.** The final ran the pipeline
  independently on each of the 30 folds the published TabFM baseline used and averaged the RMSE.
  `python judge.py val` during the run was 3-fold CV *inside* fold 0's training split, so its numbers
  (the ~1.247 above) are search-time signal, not the reported test score.
- **`python final/run.py` will not finish on this task, and the pipeline is not the reason.** After
  the TabFM pass it does `pd.read_csv(T.competition.sample_submission)` to order the submission, and
  `load_tabarena_task` never sets `T.competition` — there is no sample submission for a TabArena
  fold. Expect an `AttributeError` at that line; `T.submission(pred, T.X_test)` just above it has
  already produced the `row_id` / `scaled-sound-pressure` frame. `python final/run.py --check` stops
  before all of this and is fine. I read this off `final/judge.py:266-296` and `final/run.py:119-124`
  rather than executing it — this session had no shell — so treat it as a code reading, and note that
  `run.py` and `judge.py` are outside the two files I was allowed to write, hence left alone.
- **`folds()` is not defined**, on purpose: the rows are exchangeable within fold 0's training split
  once the configuration structure is in the features, so validation used the judge's own
  `KFold(3, shuffle, seed 0)`, whose 2/3 train ratio matches what each TabArena final fold sees.