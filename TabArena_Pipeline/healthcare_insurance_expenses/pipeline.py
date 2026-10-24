import os

import numpy as np
import pandas as pd

TABFM_KWARGS = {'norm_methods': ['none']}

MODE = 'clean'
ALPHA = 1.0        # weight on TabFM's own output vs the analytic base
SPIKE_T = 500.0    # residual above which a training row counts as a spike, in currency units
NDRAWS = 12         # independent contexts the judge averages; 1 = one draw of every row
FRAC = 0.9         # rows kept per draw when NDRAWS > 1
DUP = 1            # copies of every context row: a denser context for the same information
FEATS = 'base'      # columns reaching TabFM: 'base' | 'raw' | 'nobase' | 'all'
OFF_SHRINK = 0.05       # 0 | 'auto' | float: pull E[spike] toward 0 by its estimation variance
BASE_COL = 'base'
# postprocess is handed the *original* X_test, not the engineered one, and each hook stage runs in a
# fresh interpreter, so what preprocess fitted travels between the two stages through the workspace.
CACHE = os.path.abspath('.base_cache.npz')  # the judge chdirs into the workspace


def _design(d):
  """The closed form's own columns."""
  age = d['age'].astype(float).to_numpy()
  bmi = d['bmi'].astype(float).to_numpy()
  s = (d['smoker'] == 'yes').astype(float).to_numpy()
  cols = [age ** 2, age, d['children'].astype(float).to_numpy(), bmi,
          (d['sex'] == 'male').astype(float).to_numpy()]
  for r in ('northeast', 'northwest', 'southeast'):
    cols.append((d['region'] == r).astype(float).to_numpy())
  cols += [s, s * bmi, s * (bmi > 30).astype(float)]
  return np.column_stack(cols)


def _fit_base(X_train, y, X_test):
  """LAD hyperplane through the clean rows: the spikes are one-sided, so the median ignores them."""
  A, B = _design(X_train), _design(X_test)
  mu, sd = A.mean(0), A.std(0) + 1e-9
  A, B = np.c_[(A - mu) / sd, np.ones(len(A))], np.c_[(B - mu) / sd, np.ones(len(B))]
  try:
    from sklearn.linear_model import QuantileRegressor
    q = QuantileRegressor(quantile=0.5, alpha=0.0, solver='highs').fit(A[:, :-1], y)
    return q.predict(A[:, :-1]), q.predict(B[:, :-1])
  except Exception:                            # keep the 30 final folds alive on any solver hiccup
    c = np.linalg.lstsq(A, y, rcond=None)[0]   # trimmed least squares as a stand-in median fit
    for _ in range(40):
      r = y - A @ c
      keep = r <= np.quantile(r, 0.6)
      c = np.linalg.lstsq(A[keep], y[keep], rcond=None)[0]
    return A @ c, B @ c


def preprocess(X_train, y_train, X_test):
  Xtr, Xte = X_train.copy(), X_test.copy()
  y = np.asarray(y_train, dtype=float).ravel()
  btr, bte = _fit_base(X_train, y, X_test)
  resid = y - btr
  off = float(resid.mean())                            # E[spike]: the only thing left to estimate
  if OFF_SHRINK:                                       # that mean is over a 9%/91% mixture, so it is
    se2 = float(resid.var(ddof=1)) / len(resid)        # noisy; shrink it by its own share of variance
    s = off ** 2 / (off ** 2 + se2) if OFF_SHRINK == 'auto' else 1.0 - float(OFF_SHRINK)
    off *= s

  shift = 0.0 if MODE in ('clean', 'winsor') else off              # in 'clean' mode the context has no spikes,
  Xtr[BASE_COL] = btr + shift                          # so the column must stay on the clean scale
  Xte[BASE_COL] = bte + shift
  np.savez(CACHE, base=bte + shift, off=off)

  if MODE == 'resid':
    y_train = pd.Series(resid - off, index=y_train.index, name=getattr(y_train, 'name', 'y'))
  elif MODE == 'clean':
    keep = resid <= SPIKE_T
    if keep.sum() >= 50:                               # never hand TabFM an empty context
      Xtr, y_train = Xtr[keep], y_train[keep]
  elif MODE == 'winsor':                               # keep the spike rows' covariates, pull their
    y_train = pd.Series(np.minimum(y, btr + SPIKE_T),  # target down onto the clean surface
                        index=y_train.index, name=getattr(y_train, 'name', 'y'))
  return Xtr, y_train, Xte


RAW = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
# Every column below is a term of the closed form: dropping them costs 60 RMSE ('rawonly'), which is
# how much of the surface TabFM cannot reconstruct from the six raw columns on its own.
SETS = {
    'rawonly': RAW,
    'base': ['base'],
    'raw': RAW + ['base'],
    'nobase': RAW + ['age2', 'sm_ob', 'sm_bmi'],
    'all': RAW + ['age2', 'sm_ob', 'sm_bmi', 'base'],
    'split': RAW + ['age2', 'smk', 'ob_smk', 'sm_bmi'],
    'lin': ['age2', 'children', 'bmi', 'male', 'ne', 'nw', 'se', 'smk', 'sm_bmi', 'ob_smk'],
    'linraw': RAW + ['age2', 'children', 'male', 'ne', 'nw', 'se', 'smk', 'sm_bmi', 'ob_smk'],
    'base_eng': ['base', 'age2', 'sm_ob', 'sm_bmi'],
    'base_eng2': ['base', 'age2', 'sm_ob', 'sm_bmi', 'smoker', 'children'],
    'wide': RAW + ['age2', 'sm_ob', 'sm_bmi', 'smk', 'ob_smk', 'ob', 'male', 'age3'],
}


def engineer(X_train, y_train, X_test):
  Xtr, Xte = X_train.copy(), X_test.copy()
  for d in (Xtr, Xte):
    age = d['age'].astype(float)
    bmi = d['bmi'].astype(float)
    sm = (d['smoker'] == 'yes').astype(float)
    ob = (bmi > 30).astype(float)
    d['age2'] = age ** 2
    d['age3'] = age ** 3
    d['sm_ob'] = sm + sm * ob
    d['sm_bmi'] = sm * bmi
    d['smk'] = sm
    d['ob'] = ob
    d['ob_smk'] = sm * ob
    d['male'] = (d['sex'] == 'male').astype(float)
    for r, c in (('northeast', 'ne'), ('northwest', 'nw'), ('southeast', 'se')):
      d[c] = (d['region'] == r).astype(float)
  keep = set(SETS.get(FEATS, SETS['all']))
  cols = [c for c in Xtr.columns if c in keep or c.endswith('_id')]
  return Xtr[cols], Xte[cols]


def sample(X_train, y_train, X_test, max_rows):
  n = len(X_train)
  if DUP > 1:
    return [np.repeat(np.arange(n), DUP)]
  if NDRAWS <= 1 or FRAC >= 1.0:
    return [np.arange(n)]
  k = max(50, int(round(n * FRAC)))
  rng = np.random.default_rng(12345)
  return [np.sort(rng.choice(n, k, replace=False)) for _ in range(NDRAWS)]


def folds(X_train, y_train):
  """Two shuffled 10-folds instead of the judge's single 3-fold.  Every candidate left is inside a
  couple of RMSE units of the others, so the split has to be measured twice to rank them, and 803
  context rows per fold sits closer to the 892 the final actually fits on."""
  from sklearn.model_selection import KFold
  out = []
  for seed in (0, 1):
    out += list(KFold(10, shuffle=True, random_state=seed).split(X_train))
  return out


def postprocess(pred, y_orig, X_test):
  z = np.load(CACHE)
  base, off = z['base'], float(z['off'])
  assert len(base) == len(pred), 'base cache out of sync with the prediction'
  p = np.asarray(pred, dtype=float).ravel()
  if MODE == 'resid':
    out = base + ALPHA * p
  elif MODE in ('clean', 'winsor'):
    out = ALPHA * p + (1.0 - ALPHA) * base + off
  else:
    out = ALPHA * p + (1.0 - ALPHA) * base
  return pd.Series(out, index=pred.index, name=getattr(pred, 'name', None))
