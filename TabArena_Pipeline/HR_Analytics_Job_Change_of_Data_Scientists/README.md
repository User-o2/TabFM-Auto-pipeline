# HR Analytics — Job Change of Data Scientists (metric: 1-auc, lower is better)

## What the pipeline does

TabFM does all the predicting; the pipeline only decides how the table is
represented to it. The raw HR table is mostly strings, several of which are
ordinal but would be coded arbitrarily by TabFM's "appearance-order" categorical
encoder, so each of those is rewritten as a number that respects its natural
order: `experience` (`<1` → 0, `>20` → 21), `last_new_job` (`never` → 0,
`>4` → 5), `education_level` (Primary School … Phd → 0…4), and `company_size`
as an equal-spaced bucket rank over the eight size bands. `relevent_experience`
becomes a 0/1 flag, `training_hours` is log1p-ed (it is strongly right-skewed),
and `city`, `gender`, `enrolled_university`, `major_discipline`,
`company_type` stay categorical. Two missingness features are added, because
absent employer fields carry signal here: a count of NaNs across the eight
optional columns, and a flag for "no company_size and no company_type at all".
On top of that, TabFM's own ensemble knobs are turned up — square-root-many
random feature crosses, categorical permutation across ensemble members, and
all five per-member normalisations (`none`, `power`, `quantile`,
`quantile_rtdl`, `robust`). Each of those was measured one at a time during the
search; together they took 3-fold CV from 1-auc 0.19745 to 0.19583. Predictions
are returned unmodified (`postprocess` is the identity); no row sampling and no
`max_num_rows` override, so TabFM attends over the whole ~19k-row training
table.

The module also carries a `FEATS` dict of toggles for features that were tried
and rejected — out-of-fold target encoding of `city`, city frequency counts,
pairwise interactions, per-column NaN flags, "missing as its own level",
dropping `city`. Only `miss` and `size_rank` are `True` in the scored version;
the rest is dead-but-documented search history and the `_target_encode` helper
it needs. Leaving them in changes nothing about what runs.

## What build.sh builds

Nothing. `final/build.sh` is a no-op that exits 0: the pipeline reads no cache,
no auxiliary file, no precomputed artifact — only the DataFrames the judge
passes in. It takes no time and no disk. Run it anyway if you like; the real
command is:

```
cd <this workspace> && bash final/build.sh && python final/run.py
```

`python final/run.py` needs a GPU with TabFM available and runs 8 ensemble
members over the full training context; it writes `final/submission.csv`.
`python final/run.py --check` validates prerequisites and imports without
touching TabFM.

## Things a reader could get wrong

- `_plain()` is not cosmetic. Under pandas 3 the string columns come back as
  pyarrow-backed `StringDtype`, and the judge unpickles the hook output in a
  process that cannot always import pyarrow; `_plain` casts those columns *and
  the column index* back to plain object. Removing it breaks the run in a way
  that has nothing to do with modelling.
- `row_id` is deliberately passed through in `_feat`, but guarded by a
  membership test: it is absent under `judge.py check`, which feeds a 200-row
  slice without the id column.
- `preprocess` returns `y_train.astype(object)` — the labels stay the original
  `Yes`/`No` strings for TabFM. The 0/1 form (`yb`) is derived inside
  `engineer` only to feed the (currently disabled) target encoder.
- `TE_FOLDS` / `TE_PRIOR` / `_target_encode` are unused while
  `FEATS['city_te']` is `False`. Do not read their presence as evidence that
  the scored model used target encoding — it did not.