# Marketing_Campaign — TabFM pipeline (history/pipeline_056.py)

## What the pipeline does

The predictor is the frozen TabFM foundation model doing in-context learning over the training rows;
everything here is about what it gets to see and how its votes are combined. The task is small (1493
train / 747 test rows for the scored split), so no row selection is needed — every training row is
always in context. Three things are layered on top of that.

**Feature expansion.** `engineer` widens the 25-ish raw customer columns to ~81 by adding the ratios
and totals that actually carry the marketing-response signal: total spend and total purchases, the
per-category spend shares and log-spends, channel shares (web / catalog / store), deal share, web
conversion (web purchases per visit), spend per purchase, spend over income, income per household
member, and a spend-per-day rate. `Dt_Customer` is converted to a tenure in days from 2012-01-01
(and reused in place of the raw date), `Year_Birth` to an age, and the two categorical columns are
given ordinal encodings (`Education` → 0–4, `Marital_Status` → a has-partner flag). A short list of
the strongest columns (`DUP`) is duplicated verbatim. That is deliberate and not a bug: TabFM's
internal SVD/feature-cross budget is set as `sqrt(n_columns)`, so widening the table from 64 to 81
columns buys one more component (k: 8 → 9), and paying for it in copies of columns that already
matter beat paying for it in new, weaker ones.

**Ensembling by draw.** The judge fixes TabFM's `random_state`, so two ensemble members over the same
rows in the same order are identical. `sample` therefore returns four index arrays over the *same*
full training set — the identity permutation plus three shuffles from a fixed seed — which is what
makes the four draws differ at all. Their predictions are averaged.

**Sharpened voting.** `TABFM_KWARGS` averages probabilities rather than logits at
`softmax_temperature=0.2`, with `n_svd_features` / `n_feature_crosses` at `'sqrt'`, outlier clipping
at 2.0 σ, and categorical permutation on. The sharp temperature and the extra draws interact: more
voters let the temperature go lower before ties stop resolving.

Validation used 5-fold × 6-repeat stratified CV (`folds()`), which is far more stable than the
default 3 folds on a dataset this small — most of the tuning steps were worth less than the fold
spread otherwise.

Scored result: **1-AUC = 0.0616** as the mean over the 30 TabArena folds (published TabFM baselines
on the same folds: 0.0730 plain, 0.0724 ensemble). Search-time CV best was 0.0703.

## What build.sh builds

Nothing. `bash final/build.sh` prints one line and exits 0 in well under a second, needs no disk, and
exists only so the documented `bash final/build.sh && python final/run.py` sequence works. The
pipeline has no prerequisites: all four hooks are pure functions of the DataFrames the judge passes
in, and the run that produced them never wrote a cache, a feature file, or a helper module. (This is
also why there is no `final/build/` directory — there were no build scripts to copy into it.)

To reproduce, you need the competition data in place under `data/` and a GPU with TabFM available;
`python final/run.py` runs the same `judge.load_task` / `judge.run_pipeline` path used during the
search. `python final/run.py --check` verifies imports and prerequisites without touching TabFM.

## Things a reader could get wrong

- **The duplicated columns in `DUP` are intentional.** Deleting them changes TabFM's `sqrt`-sized
  SVD/cross budget and costs accuracy; they are not leftover debris.
- **`sample` is not subsampling.** It returns four permutations of *all* training rows. If you
  "simplify" it to a single `np.arange(n)`, the four ensemble draws collapse into one identical
  prediction, because the judge pins `random_state`.
- **`MARITAL` is dead code.** An ordinal marital encoding that lost to the binary `HasPartner` flag;
  the dict is still defined but never used.
- **Row counts differ between the run record and TabArena.** `result.json` reports 1493/747 for the
  search split, but the 30-fold final ran 44800 train / 22400 test rows in total across folds.
- **Hard-coded dates.** Tenure is measured from 2012-01-01 and age from 2014, matching this
  dataset's collection window; they are constants, not "today".
- **The docstring at the top of `pipeline.py` describes the *next* experiment** (eval 56's intent),
  as the search loop required. It is the record of why this configuration was tried, not a
  description of a pending change.