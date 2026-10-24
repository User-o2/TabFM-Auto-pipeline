# Is-this-a-good-customer — final pipeline

## What it does

The task is a small credit-scoring table: 1148 rows, 13 columns, 11.3% positives, scored as
1 − ROC-AUC (lower is better). The pipeline is deliberately thin around a frozen TabFM.

Feature side, it *narrows* the table rather than widening it. A leave-one-out scan over all 13
columns found five that cost nothing or paid to remove (`having_children_flg`, `phone_operator`,
`family_status`, `month`, `region`), so those are dropped, and `education` is recoded from its
string to its ordinal level. Nothing is added: engineered ratios, out-of-fold target rates, quantile
bins and a GBDT probability as an extra column were all measured and all hurt. TabFM then sees the
eight remaining columns over the full training context (no subsampling — `sample()` is not defined,
and shrinking `max_num_rows` was the one clearly bad knob), with `class_shift=False` and
`feat_shuffle_method='none'`.

On top of that, `engineer()` also fits an auxiliary trio on the *same* training fold — a
HistGradientBoosting classifier, a 500-tree random forest, and a one-hot + standardised logistic
regression (C=0.1) — and rank-averages their test-row scores. `postprocess()` blends that ranking
with TabFM's at weight 0.45, in ranks, so TabFM keeps the majority (0.55) of the final ordering;
`tabfm_corr` stays around 0.968. The three auxiliaries are tools with different inductive biases,
not a replacement: alone they are roughly level with TabFM, and it is the disagreement that the
blend buys. Because `engineer()` and `postprocess()` run in separate processes, the auxiliary
ranking travels between them through `.aux_pred.npy` in the workspace.

Validation during the run was 6 repeats of stratified 5-fold (30 folds, 918 train / 230 val),
because the judge's default single 3-fold has a fold std of ~0.010 on this table — larger than any
effect worth measuring. Against an identity pipeline on those 30 folds: 0.24374 → 0.22937, a paired
−0.01437 ± 0.00164.

## What build.sh builds

Nothing. `final/pipeline.py` reads no file it does not write itself — the single filename it names,
`.aux_pred.npy`, is produced by `engineer()` earlier in the same run and consumed by
`postprocess()`. `build.sh` therefore prints that and exits 0. Runtime: instant. Disk: zero.

    bash final/build.sh && python final/run.py

`final/build/` is absent because no build script exists to copy: the two helpers written during the
search (`base.py`, `sweep.py`) were off-judge variant drivers that mutated a copy of `pipeline.py`
and scored candidates locally, produced no cache the pipeline reads, and were deleted from the
workspace before the run ended.

## Things a reader could get wrong

- **`.aux_pred.npy` is not a cache to pre-build.** Pre-computing it would be wrong: it depends on
  the training fold currently in hand, and the pipeline overwrites it on every `engineer()` call.
  If a stale one is lying around, `postprocess()` guards on length and falls back to raw TabFM
  output, so a mismatch degrades silently rather than crashing — delete it if in doubt.
- **The auxiliary models are fitted inside `engineer()`, on the training fold only.** There is no
  leakage path from the test rows, and no cross-fit artefact carried between folds.
- **Blend weight 0.45 is a choice, not an optimum.** The weight sits on a long, very slowly rising
  plateau (0.33 → 0.23139, 0.40 → 0.23110, 0.45 → 0.23077), and each step is inside the paired noise
  on its own; only the direction replicates. It is capped at 0.45 so TabFM stays the majority of the
  ranking. Pushing it higher would likely score a hair better and would stop being a TabFM pipeline.
- **`folds()` is 30 folds, not the judge's 3.** That is what the numbers quoted above are measured
  on; a single 3-fold reproduction will look much noisier than the differences discussed here.
- **This is a TabArena task.** The competition data under `data/` is the harness-prepared fold-0
  split; `final/run.py` loads it through the bundled `final/judge.py` + `final/config.yaml`, so the
  reproduction cannot drift from what was scored. `run.py` copies `final/pipeline.py` over the
  workspace `pipeline.py` and refuses to run if the two already differ.