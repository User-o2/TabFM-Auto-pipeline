import numpy as np

VARIANT = 'base'

TABFM_KWARGS = {'norm_methods': ['power', 'quantile_rtdl']}

_DROP = ['ghazard', 'nbumps5', 'seismoacoustic', 'maxenergy', 'nbumps2', 'nbumps3', 'nbumps4', 'energy']
_ADD = ['log_gpp', 'tranks']


def folds(X_train, y_train):
  from sklearn.model_selection import RepeatedStratifiedKFold
  y = np.asarray(y_train).ravel()
  rskf = RepeatedStratifiedKFold(n_splits=3, n_repeats=10, random_state=20260908)
  return [(tr, va) for tr, va in rskf.split(np.zeros(len(y)), y)]


def preprocess(X_train, y_train, X_test):
  return X_train, y_train, X_test


def _shape(X):
  X = X.copy()
  if 'g_per_pulse' in _ADD:
    X['g_per_pulse'] = X['genergy'] / X['gpuls'].clip(lower=1.0)
  if 'log_gpp' in _ADD:                # same ordering as g_per_pulse, different spacing
    X['g_per_pulse'] = np.log1p(X['genergy'] / X['gpuls'].clip(lower=1.0))
  # gdenergy/gdpuls are percent deviations of genergy/gpuls from the seam's running
  # average, so their ratio is the deviation of the energy-per-pulse that g_per_pulse
  # measures in absolute terms.
  if 'gd_ratio' in _ADD:
    X['gd_ratio'] = (100.0 + X['gdenergy']) / (100.0 + X['gdpuls']).clip(lower=5.0)
  if 'gd_gap' in _ADD:
    X['gd_gap'] = X['gdenergy'] - X['gdpuls']
  if 'g_avg' in _ADD:
    X['g_avg'] = X['genergy'] / (1.0 + X['gdenergy'] / 100.0).clip(lower=0.05)
  if 'seismic_shift' in _ADD:
    X['seismic_shift'] = X['seismic'].astype(str) + X['shift'].astype(str)
  if 'ordcat' in _ADD:
    X['seismic'] = X['seismic'].astype(str).map({'a': 0.0, 'b': 1.0, 'c': 2.0, 'd': 3.0})
    X['shift'] = X['shift'].astype(str).map({'N': 0.0, 'W': 1.0})
  if 'dupcols' in _ADD:                # a strong column twice: more of the permuted views see it
    X['nbumps_2'] = X['nbumps']
    X['gpuls_2'] = X['gpuls']
  if 'nb_pos' in _ADD:
    X['nb_pos'] = (X['nbumps'] > 0).astype(float)
  if 'logs' in _ADD:                   # raw-scale members ('none') see a compressed table
    for c in ('genergy', 'gpuls'):
      X[c] = np.log1p(X[c])
    X['g_per_pulse'] = np.log1p(X['g_per_pulse'])
  return X.drop(columns=[c for c in _DROP if c in X.columns])


def _rankify(A, B, gauss=False, min_unique=0, pooled=False):
  """Replace numeric columns by their percentile within the training fold."""
  A, B = A.copy(), B.copy()
  for c in A.columns:
    if A[c].dtype.kind not in 'ifu' or A[c].nunique() < min_unique:
      continue
    ref = np.sort(np.concatenate([A[c].to_numpy(float), B[c].to_numpy(float)])
                  if pooled else A[c].to_numpy(float))
    for F in (A, B):
      p = np.searchsorted(ref, F[c].to_numpy(float), side='left') / len(ref)
      if gauss:
        from scipy.special import ndtri
        p = ndtri(np.clip(p, 0.5 / len(ref), 1 - 0.5 / len(ref)))
      F[c] = p
  return A, B


def engineer(X_train, y_train, X_test):
  A, B = _shape(X_train), _shape(X_test)
  if 'ranks' in _ADD:
    A, B = _rankify(A, B)
  if 'granks' in _ADD:
    A, B = _rankify(A, B, gauss=True)
  if 'ranks_hi' in _ADD:
    A, B = _rankify(A, B, min_unique=20)
  if 'granks_hi' in _ADD:
    A, B = _rankify(A, B, gauss=True, min_unique=20)
  if 'tranks' in _ADD:                 # reference distribution pooled over train+test (no labels)
    A, B = _rankify(A, B, gauss=True, pooled=True)
  if 'tranks_u' in _ADD:
    A, B = _rankify(A, B, pooled=True)
  return A, B


def postprocess(pred, y_orig, X_test):
  return pred
