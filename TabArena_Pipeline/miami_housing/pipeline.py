import numpy as np

TABFM_KWARGS = {'n_feature_crosses': 'sqrt',
                # a cycle over members: quantile and quantile_rtdl on the plain (even) members,
                # quantile_rtdl and power on the crossed (odd) ones. Order is load-bearing.
                'norm_methods': ['quantile', 'quantile_rtdl', 'quantile_rtdl', 'power'],
                'outlier_threshold': 20.0}

# every non-negative column: LATITUDE would collapse to a constant and LONGITUDE is negative
LOGGED = ['LND_SQFOOT', 'TOT_LVG_AREA', 'SPEC_FEAT_VAL', 'RAIL_DIST', 'OCEAN_DIST', 'WATER_DIST',
          'CNTR_DIST', 'SUBCNTR_DI', 'HWY_DIST', 'age', 'month_sold', 'structure_quality',
          'avno60plus']


def folds(X_train, y_train):
  """Three repeats of 3-fold: nine splits, each training on 2/3 of the rows like the final does."""
  from sklearn.model_selection import RepeatedKFold
  return list(RepeatedKFold(n_splits=3, n_repeats=3, random_state=0).split(X_train))


def preprocess(X_train, y_train, X_test):
  return X_train, y_train, X_test


def _log(df):
  """Invisible to the rank-based members, and the whole point for the crossed ones."""
  d = df.copy()
  for c in LOGGED:
    if c in d.columns:
      d[c] = np.log1p(np.maximum(d[c].to_numpy(dtype=float), 0.0))
  return d


SHIFT = 1024.0


def engineer(X_train, y_train, X_test):
  A, B = _log(X_train), _log(X_test)
  cols = [c for c in LOGGED if c in A.columns]
  off = SHIFT * A[cols].std()
  A[cols], B[cols] = A[cols] + off, B[cols] + off
  return A, B


def postprocess(pred, y_orig, X_test):
  return pred
