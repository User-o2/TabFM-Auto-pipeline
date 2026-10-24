# heloc — final pipeline

## What it does

HELOC (FICO) credit-risk classification: 23 integer bureau columns, 6972 training rows, scored as
1 − AUC (lower is better). The prediction is made entirely by the frozen TabFM in-context model; the
pipeline is a thin, deliberately minimal wrapper around it and the wrapper's whole content is three
decisions.

First, the feature view. TabFM is run with a single normalisation, `norm_methods=["quantile"]`,
instead of the shipped `["none", "power"]` mix. Quantile-normalising pushes each column's ranks
through the normal quantile function, and measured on its own that view beats every other view and
every mix containing it — mixing only averages the weaker views back in. `class_shift=False` is the
one other knob that helped.

Second, the context. `sample()` returns one draw of *every* training row, ordered by
`ExternalRiskEstimate`. Every variant that removed rows lost roughly in proportion to how many went
(dropping the 396 "no bureau record" rows alone cost +0.0088, because 5.5% of query rows are that
same point and without it in context they have no evidence), and adding synthetic interpolated rows
was by far the worst result of the run (+0.015 at 50% synthetic). The ordering is nearly free —
attention over the context is permutation-invariant — and is kept only because it costs nothing.

Third, one derived column. `MSinceMostRecentInqexcl7days` codes "no inquiries at all" as −8 (a good
risk) and "an inquiry in the last 7 days" as −7 (a bad one), so under a rank view its best and worst
cases both sort below every real month count. `engineer()` adds `inq_mono`, a re-ordered copy that
places −8 above every month count, −7 just below zero and −9 at the bottom, *beside* the raw column
rather than replacing it. FICO's three sentinels are otherwise left exactly as written — blanking
them costs +0.0044. `preprocess` and `postprocess` are identity, so TabFM's probability ranking is
the submitted ranking.

`folds()` defines 3× repeated stratified 5-fold (15 fits) so that validation is stable enough to
resolve the ~0.0002 effects the run was chasing; it does not affect the final test prediction.

Validation score on those folds: 0.19463 (identity pipeline: 0.19591).

## What build.sh builds

Nothing. `final/pipeline.py` reads no files — no caches, no fitted artefacts, no precomputed
features. `build.sh` is a comment and `exit 0`; it takes no time and no disk. `final/build/` is
empty because no standalone scripts were written during the run: every experiment was a direct edit
of `pipeline.py` followed by `python judge.py val`.

    bash final/build.sh && python final/run.py

## Things a reader might otherwise get wrong

- **The docstring numbers are paired deltas, not absolute scores.** Every `+0.000xx` in
  `pipeline.py` is the cost of a variant against this configuration on the same 15 folds. Positive
  means worse (the metric is 1 − AUC).
- **The mixed default is not a diversity win here.** The natural instinct is to keep several
  normalisation views for ensemble diversity; on this task that is strictly worse than the single
  quantile view, because `none` is half the default mix and is the weakest view of the five.
- **`inq_mono` is added, not substituted.** Overwriting the raw column with the recode costs
  +0.00127 — TabFM wants both the sentinel's identity and the corrected ordering. Adding further
  recodes of the same kind also loses; the plateau of useful derived columns tops out very early.
- **Other models were measured and rejected.** HistGradientBoosting (0.20204), extra trees, random
  forest and quantile+logistic all trail TabFM (0.19519 on the same folds); blending buys ~0.0004 and
  feeding one in as an out-of-fold column costs +0.0042. None of that is in the shipped pipeline.
- **`run.py` copies `final/pipeline.py` over the workspace `pipeline.py`** and refuses to run if the
  two differ. They are identical as shipped; if you edit one, reconcile both.