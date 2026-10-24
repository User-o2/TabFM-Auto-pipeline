import numpy as np
import pandas as pd

TABFM_KWARGS = {'max_num_rows': 65536, 'n_svd_features': 'sqrt'}   # above the row count: every row stays in context

CUT = {'Fair': 0, 'Good': 1, 'Very Good': 2, 'Premium': 3, 'Ideal': 4}
COLOR = {'J': 0, 'I': 1, 'H': 2, 'G': 3, 'F': 4, 'E': 5, 'D': 6}
CLARITY = {'I1': 0, 'SI2': 1, 'SI1': 2, 'VS2': 3, 'VS1': 4, 'VVS2': 5, 'VVS1': 6, 'IF': 7}

# preprocess and postprocess run in separate subprocesses over a shared workspace,
# so the fold's test-row baseline travels through a scratch file, not a global.
_STASH = '.base_test.npy'
SHRINK = 1.043    # weight on TabFM's correction; solved for, flat over 1.00-1.10
_GUARD = 0.45     # five conditional sigmas: the correction cannot exceed this share of base

# knots where the trade itself puts them: the "magic" carat weights buyers pay a
# premium to cross (0.3, 0.5, 0.7, 1.0, 1.5, 2.0) plus fill-in knots between them.
_KNOTS = np.log(np.array([0.3, 0.4, 0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]))
_RIDGE, _KFOLD = 1e-5, 5
_BOUNDS = (250.0, 25000.0)   # the trend's output range; set per fold from the prices seen


_GRADES = (('cut', 5), ('color', 7), ('clarity', 8))
_RANK_CODES = False   # True = uniform spacing, only rescaled; False = spaced by fitted value
_CODES = {}        # per-fold grade codes, shared from preprocess to engineer


def _feat(d, codes=None):
  """Domain features. `codes` respaces the grade columns; the ridge always gets the
  plain 0..k-1 ranks, because it builds its dummies by comparing against them."""
  o = pd.DataFrame(index=d.index)
  ct = d['carat'].astype(float)
  o['carat'] = ct
  o['cut'] = d['cut'].map(CUT).astype(float)
  o['color'] = d['color'].map(COLOR).astype(float)
  o['clarity'] = d['clarity'].map(CLARITY).astype(float)
  if codes:
    for name, _ in _GRADES:
      o[name] = codes[name][o[name].to_numpy().astype(int)]
  o['depth'] = d['depth'].astype(float)
  o['table'] = d['table'].astype(float)
  x = d['x'].replace(0.0, np.nan).astype(float)      # a zero dimension is missing, not small
  y = d['y'].replace(0.0, np.nan).astype(float)
  z = d['z'].replace(0.0, np.nan).astype(float)
  o['x'], o['y'], o['z'] = x, y, z

  o['vol'] = x * y * z
  o['log_carat'] = np.log(ct)
  o['density'] = ct / (x * y * z)
  o['size'] = np.cbrt(ct)
  o['diam'] = 0.5 * (x + y)
  o['yx'] = y / x
  o['zx'] = z / x
  o['depth_calc'] = 200.0 * z / (x + y)
  o['table_mm'] = o['table'] * 0.005 * (x + y)
  o['depth_dev'] = np.abs(o['depth'] - 61.9)
  o['table_dev'] = np.abs(o['table'] - 56.5)
  return o


def _design(f):
  """The trend in log-price: a carat spline, the three grades, and their pairs.
  Cells are enumerated, never support-filtered, so train and test always produce
  the same 259 columns."""
  lc = f['log_carat'].to_numpy()
  cut, color, clarity = (f[c].to_numpy() for c in ('cut', 'color', 'clarity'))
  cols = [np.ones_like(lc), lc, lc ** 2, lc ** 3]
  cols += [np.maximum(lc - k, 0.0) for k in _KNOTS]

  for g, k in ((cut, 5), (color, 7), (clarity, 8)):
    for j in range(1, k):                              # drop first level, keep intercept
      d = (g == j).astype(float)
      cols += [d, d * lc, d * lc ** 2]
  for A, ka, B, kb in ((color, 7, clarity, 8), (cut, 5, clarity, 8), (cut, 5, color, 7)):
    for a in range(1, ka):
      for b in range(1, kb):
        d = ((A == a) & (B == b)).astype(float)
        cols += [d, d * lc]

  dp, tb = f['depth'].to_numpy() - 61.75, f['table'].to_numpy() - 57.46
  cols += [dp, dp ** 2, tb, tb ** 2, dp * tb]
  return np.nan_to_num(np.column_stack(cols), nan=0.0, posinf=0.0, neginf=0.0)


def _grade_codes(f, t):
  """Recode each grade level by its additive effect on log-price, order preserved."""
  lc = f['log_carat'].to_numpy()
  cols, spans = [np.ones_like(lc), lc, lc ** 2, lc ** 3], {}
  for name, k in _GRADES:
    spans[name] = (len(cols), k)
    cols += [(f[name].to_numpy() == j).astype(float) for j in range(1, k)]
  A = np.nan_to_num(np.column_stack(cols), nan=0.0)
  reg = 1e-8 * A.shape[0] * np.eye(A.shape[1])
  reg[0, 0] = 0.0
  w = np.linalg.solve(A.T @ A + reg, A.T @ t)
  out = {}
  for name, k in _GRADES:
    i, _ = spans[name]
    e = np.arange(float(k)) if _RANK_CODES else np.concatenate([[0.0], w[i:i + k - 1]])
    out[name] = (e - e.mean()) / (e.std() + 1e-9)      # centred, unit scale, monotone
  return out


def _solve(A, t):
  reg = _RIDGE * A.shape[0] * np.eye(A.shape[1])
  reg[0, 0] = 0.0
  w = np.linalg.solve(A.T @ A + reg, A.T @ t)
  e = np.exp(A @ w)
  scale = float(e @ np.exp(t) / (e @ e))   # least squares in dollars, not Duan's mean-of-ratios
  return w, scale


def _base(A, w, scale):
  return np.clip(scale * np.exp(A @ w), *_BOUNDS)


def _oof_base(A, t):
  """The training rows' own baseline, each fifth of them fit without themselves."""
  out = np.empty(len(t))
  edge = np.linspace(0, len(t), _KFOLD + 1).astype(int)
  order = np.random.default_rng(0).permutation(len(t))
  for i in range(_KFOLD):
    va = order[edge[i]:edge[i + 1]]
    tr = np.setdiff1d(order, va, assume_unique=True)
    out[va] = _base(A[va], *_solve(A[tr], t[tr]))
  return out


def preprocess(X_train, y_train, X_test):
  global _BOUNDS
  f_tr, f_te = _feat(X_train), _feat(X_test)
  t = np.log(np.asarray(y_train, dtype=float))
  _BOUNDS = (float(np.min(y_train)), float(np.max(y_train)))
  _CODES.update(_grade_codes(f_tr, t))
  A_tr = _design(f_tr)
  base_tr = _oof_base(A_tr, t)
  base_te = _base(_design(f_te), *_solve(A_tr, t))
  np.save(_STASH, base_te)
  X_train, X_test = X_train.copy(), X_test.copy()
  X_train['base'], X_test['base'] = base_tr, base_te

  return X_train, y_train.astype(float) - base_tr, X_test


def engineer(X_train, y_train, X_test):
  a, b = _feat(X_train, _CODES), _feat(X_test, _CODES)
  a['base'], b['base'] = X_train['base'].to_numpy(), X_test['base'].to_numpy()
  return a, b


def postprocess(pred, y_orig, X_test):
  base = np.load(_STASH)
  out = SHRINK * np.asarray(pred, dtype=float) + base
  out = np.clip(out, base * (1.0 - _GUARD), base * (1.0 + _GUARD))
  # price is truncated in this data, so anything outside the observed range is wrong
  return np.clip(out, float(np.min(y_orig)), float(np.max(y_orig)))
