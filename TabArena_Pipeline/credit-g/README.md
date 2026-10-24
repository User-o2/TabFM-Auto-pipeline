# credit-g — final pipeline

## What it does

The predictor is frozen TabFM, in context over all 666 training rows; the pipeline's whole job is to
change how the German-credit table is *represented* before TabFM attends over it. Ten of the thirteen
non-numeric columns encode something ordered — how much is in the checking account, how much is
saved, how long the job has lasted, how the past credits went, what property is owned, who else is
liable, what other instalment plans run, how the applicant is housed, whether there is a telephone,
whether the applicant is a foreign worker — but reach the model as unordered appearance codes.
`engineer` replaces each in place by its rank on a fixed risk ordering, so they arrive as monotone
numerics (the one place the data overrules the canonical reading is `credit_history`, where
"delay in paying off in past" is coded below "existing credits paid duly till now" to keep the column
monotone in bad rate). The three genuinely unordered columns stay categorical, with their sparse
levels merged: `credit_purpose` folds every level rarer than 35/599 of the training rows into
`other` — decided on the training rows only, expressed as a share so it scales to the real fold size —
and `personal_status_sex` merges its two smallest male levels into `male: not single`; `job` is left
untouched. `TABFM_KWARGS` then turns on every view-diversity device the model offers: per-member
permutation of the categorical codes, sqrt(n) random feature crosses and sqrt(n) SVD components for
half the members, and four normalisations cycled across the eight members instead of two.
`preprocess`, `sample` and `postprocess` are the identity — every training row stays in context and
TabFM's own ranking is what is submitted. `folds` replaces the judge's 3-fold CV with repeated
stratified 10-fold (10 repeats), because the 3-fold spread (std 0.028) is far larger than any effect
worth measuring here. On those 100 splits this pipeline scores 0.17996 (1 − ROC-AUC, lower is better)
against 0.19707 for the identity pipeline, paired t = 10.

## What build.sh builds

Nothing. `final/build.sh` is a comment and `exit 0`: the pipeline reads no files and needs no caches,
fitted artefacts or derived tables — every constant it uses is a literal in `pipeline.py`, and the
data arrives through the judge. It runs instantly and needs no disk.

Reproduce with:

    cd <this workspace> && bash final/build.sh && python final/run.py

The cost is one TabFM pass over 666 training rows with `judge.estimators_final` members; the dataset
is tiny, so this is minutes on a GPU, not hours.

## Things a reader could get wrong

- **`final/build/` is not build machinery.** Those three scripts were the run's analysis and editing
  helpers, and `build.sh` deliberately does not call them. `local.py` is a sklearn-only sanity check
  used to sketch feature ideas cheaply, `cmp.py` does paired t-tests between two `[RESULT n]` fold
  arrays in `log.txt`. `setkw.py` **rewrites `pipeline.py` in place** — running it now would edit the
  scored pipeline. All three assume the workspace root as their cwd.
- **The `credit_history` ordering is intentional, not a typo.** It contradicts the canonical Statlog
  reading; it is worth −0.0017 at t = −2.65 because it makes the column monotone in bad rate. The
  same inversion applied to `employment_since` and `savings_status` made things worse, so those keep
  the canonical order.
- **`RARE_FRAC` is a share, not a count, on purpose.** At the 599 rows a validation fold gives it is
  exactly 35 rows (the tuned value); at the 666 the final trains on it becomes ~39. The level it has
  to decide about, `education`, has ~33 ± 5 rows, so a hard-coded 35 would fold it only about two
  thirds of the time in the real folds — and not folding it costs +0.0022.
- **`engineer` casts its string columns back to `object` on the way out.** That is not cosmetic: a
  string column built here comes out arrow-backed and the judge process cannot unpickle that.
- **The scores quoted above are on this pipeline's own 100-split CV**, not on the judge's default
  3-fold and not on the held-out test split.