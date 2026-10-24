import numpy as np

TABFM_KWARGS = {'permute_categorical': True, 'n_feature_crosses': 'sqrt'}
STRONG_FRAC = 0.8        # share of flags kept numeric with a fixed target-aligned orientation


def preprocess(X_train, y_train, X_test):
  return X_train, y_train, X_test


def engineer(X_train, y_train, X_test):
  cols = [c for c in X_train.columns if c != 'row_id']
  lab = y_train.astype(str).to_numpy()
  classes = np.unique(lab)
  if len(classes) < 2:                      # `check` runs on a slice of the class-sorted train.csv
    return X_train, X_test
  y = (lab == classes[-1]).astype(int)
  B = X_train[cols].to_numpy(np.int64)
  # P(malware | flag=1) - P(malware | flag=0), guarding flags that are constant in this fold
  n1 = B.sum(axis=0)
  p1 = np.divide(B.T @ y, n1, out=np.full(len(cols), y.mean()), where=n1 > 0)
  p0 = np.divide((1 - B).T @ y, len(B) - n1, out=np.full(len(cols), y.mean()), where=n1 < len(B))
  d = p1 - p0
  strong = np.abs(d) >= np.quantile(np.abs(d), 1 - STRONG_FRAC)
  # strong flags: numeric, oriented so 1 is always the malware-associated state
  flip = [c for c, di, s in zip(cols, d, strong) if s and di < 0]
  # weak flags: categorical, so permute_categorical re-polarises them per ensemble member
  cast = {c: 'category' for c, s in zip(cols, strong) if not s}
  return (X_train.assign(**{c: 1 - X_train[c] for c in flip}).astype(cast),
          X_test.assign(**{c: 1 - X_test[c] for c in flip}).astype(cast))


def postprocess(pred, y_orig, X_test):
  return pred


def folds(X_train, y_train):
  # 2 repeats of a 10-fold: 4495 context rows per fold, the closest this training set gets to the
  # 4994 the final fits on, on the seed that took no part in choosing STRONG_FRAC
  from sklearn.model_selection import RepeatedStratifiedKFold
  cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=2, random_state=3)
  return list(cv.split(X_train, y_train))
