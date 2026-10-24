# concrete_compressive_strength — final pipeline

## What it does

The pipeline is a pure feature representation for TabFM: the frozen model makes the prediction
in-context over the training rows, with default ensemble settings (`TABFM_KWARGS = {}`), no row
sampling, no target transform and no post-processing. All of the work is in `engineer`, which
replaces the eight raw mix-design columns with a twelve-column concrete-chemistry view. Cement,
blast-furnace slag and fly ash are folded into two cementitious aggregates — the plain binder mass
and an efficiency-weighted `ceff` (slag at 1.0, fly ash at 0.4, coefficients chosen by a scan over
the published range) — and the water content is expressed as ratios rather than an absolute mass:
`water/cement` and a volumetric `water/binder` using the specific gravities 3.15 / 2.90 / 2.30.
Age enters as `log(Age)`, since strength gain is logarithmic in curing time. The two aggregate
masses become a total and a fine fraction, and superplasticizer becomes a dose relative to binder.
The five columns that are then fully carried by these derived features — CoarseAggregate,
FineAggregate, Superplasticizer, Cement and BlastFurnaceSlag — are dropped; the last two were
removed on paired evidence across repeated CV folds late in the run, each confirmed on two seeds.

Validation used `RepeatedKFold(n_splits=10, n_repeats=14, random_state=0)`, so column decisions of
this size (~0.01 RMSE) could be judged as paired differences over 140 folds rather than as a single
CV number. Rows are exchangeable here — no time or group structure — so plain repeated K-fold
mirrors the test split. Reference validation RMSE for the 12-column model: 3.72740.

## What build.sh builds

Nothing. `final/build.sh` is a comment and `exit 0`: `pipeline.py` has no prerequisites — it reads
no caches, no fitted artifacts and no derived tables, only the DataFrames the judge passes in. It
runs instantly and needs no disk.

## Things a reader could get wrong

- **`final/build/` is not a data-building directory.** `setk.py` and `cmp.py` are search-loop
  tooling (rewriting the pipeline's docstring/kwargs, and running a paired t-test over `log.txt`
  fold vectors). They are preserved for the record and are deliberately not called by `build.sh`.
- **`water_cement` divides by raw Cement, which is dropped afterwards.** The column disappears from
  the model but is still needed inside `_feat`; the drop happens last, on purpose.
- **`water_cement` and `sp_binder` can be infinite in principle.** Nothing in this dataset has zero
  cement or zero binder, so the guard was never needed and never added. A different mix table could
  produce `inf` here.
- **The docstring at the top of `pipeline.py` is the intent note for evaluation 80**, the last one
  the budget allowed — it describes the BlastFurnaceSlag removal as a live hypothesis, written
  before that evaluation ran. It is a lab-notebook entry, not a description of the final result;
  the code below it is what was scored.