import numpy as np
import pandas as pd

TABFM_KWARGS = {'n_feature_crosses': 'sqrt'}

DROP = ['OperatingSystems', 'Browser', 'Region', 'Informational', 'Informational_Duration']

NOMINAL = []

LOG1P = []

MERGE_INFO = False

CLEAN = 0.0


def folds(X_train, y_train):
  from sklearn.model_selection import RepeatedStratifiedKFold
  rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=4, random_state=12345)
  return list(rskf.split(X_train, y_train))


def _binary(y):
  """0/1 for the target however it arrives: the judge hands over 'True'/'False' strings, a local
  read_csv gives bools."""
  a = np.asarray(y).ravel()
  return (a == np.unique(a)[-1]).astype(int)


def _keep_clean(X, y, frac):
  """Indices of the rows to keep: drop, within each class, the `frac` of rows a boosted model
  fitted out-of-fold on this same training fold finds least compatible with their label."""
  from sklearn.ensemble import HistGradientBoostingClassifier
  from sklearn.model_selection import StratifiedKFold
  A = X.drop(columns=[c for c in ('row_id',) if c in X.columns]).copy()
  for c in A.columns:
    if A[c].dtype == object or str(A[c].dtype) == 'str':
      A[c] = A[c].astype('category')
    elif A[c].dtype == bool:
      A[c] = A[c].astype(int)
  z = _binary(y)
  oof = np.zeros(len(A))
  for tr, va in StratifiedKFold(4, shuffle=True, random_state=0).split(A, z):
    m = HistGradientBoostingClassifier(categorical_features='from_dtype', random_state=0,
                                       max_iter=150, learning_rate=0.1, max_leaf_nodes=16,
                                       min_samples_leaf=30, l2_regularization=1.0,
                                       early_stopping=False).fit(A.iloc[tr], z[tr])
    oof[va] = m.predict_proba(A.iloc[va])[:, 1]
  agree = np.where(z == 1, oof, 1.0 - oof)
  keep = np.ones(len(A), dtype=bool)
  for cls in (0, 1):
    idx = np.flatnonzero(z == cls)
    n = int(round(frac * len(idx)))
    if n:
      keep[idx[np.argsort(agree[idx])[:n]]] = False
  return keep


def preprocess(X_train, y_train, X_test):
  if CLEAN > 0:
    keep = _keep_clean(X_train, y_train, CLEAN)
    X_train = X_train.loc[keep].reset_index(drop=True)
    y_train = y_train.loc[keep].reset_index(drop=True)
  return X_train, y_train, X_test


def engineer(X_train, y_train, X_test):
  out = []
  for df in (X_train, X_test):
    df = df.copy()
    if MERGE_INFO and 'Informational' in df.columns:
      df['Administrative'] = df['Administrative'] + df['Informational']
      df['Administrative_Duration'] = df['Administrative_Duration'] + df['Informational_Duration']
    df = df.drop(columns=[c for c in DROP if c in df.columns])
    for c in LOG1P:
      if c in df.columns:
        df[c] = np.log1p(df[c].to_numpy(dtype=float).clip(0))
    for c in NOMINAL:
      # dtype=object, explicitly: pandas 3 makes any string column a pyarrow-backed StringDtype
      if c in df.columns:
        df[c] = pd.Series(np.array(['c%s' % v for v in df[c].to_numpy()], dtype=object),
                          index=df.index, dtype=object)
    out.append(df)
  return out[0], out[1]


def postprocess(pred, y_orig, X_test):
  return pred
