import numpy as np
import pandas as pd

TABFM_KWARGS = {'norm_methods': ['none'], 'average_logits': False}


def folds(X_train, y_train):
  from sklearn.model_selection import RepeatedStratifiedKFold
  rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0)
  return list(rskf.split(X_train, y_train))


def preprocess(X_train, y_train, X_test):
  return X_train, y_train, X_test


def _add(X):
  X = X.copy()
  w = X["Weight_in_gms"].to_numpy(dtype=float)
  c = X["Cost_of_the_Product"].to_numpy(dtype=float)
  d = X["Discount_offered"].to_numpy(dtype=float)
  det = (d > 10) | ((w > 2000) & (w < 4000)) | (w > 6000)
  lite = w <= 2000
  pocket = lite & (c <= 200)
  near = (w >= 4000) & (w <= 4200)
  seg = np.where(det, "det", np.where(pocket, "pocket",
                 np.where(lite, "lite", np.where(near, "near", "heavy"))))
  # plain object dtype: the judge host has no pyarrow, so a pyarrow-backed
  # pandas string column cannot be unpickled on the other side of the hook.
  X["seg"] = pd.Series(seg, index=X.index, dtype=object)
  return X[KEEP]


KEEP = ["Cost_of_the_Product", "seg"]


def engineer(X_train, y_train, X_test):
  return _add(X_train), _add(X_test)


def sample(X_train, y_train, X_test, max_rows):
  n = len(X_train)
  rng = np.random.default_rng(0)
  k = int(round(0.9 * n))
  return [np.sort(rng.choice(n, k, replace=False)) for _ in range(6)]


def postprocess(pred, y_orig, X_test):
  return pred
