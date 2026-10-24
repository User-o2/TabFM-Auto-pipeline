import numpy as np
import pandas as pd

TABFM_KWARGS = {
    "norm_methods": ["none", "power", "quantile", "robust"],
    "average_logits": False,
    "enable_nnls": True,
    "nnls_beta": 1.0,
    "n_svd_features": "sqrt",
    "softmax_temperature": 1.4,
    "permute_categorical": True,
}


def preprocess(X_train, y_train, X_test):
  return X_train, y_train, X_test


MERGE = {"hld": ["housing", "loan", "default"]}


def _feat(X):
  X = X.copy()
  for new, cols in MERGE.items():
    if all(c in X.columns for c in cols):
      vals = [X[c].fillna("").astype(str).to_numpy(dtype=object) for c in cols]
      # object dtype on purpose: the judge server that unpickles the hook output has no pyarrow, so a
      # freshly created pandas-3 `str` column (arrow-backed inside the hook sandbox) cannot be
      X[new] = pd.Series(["|".join(t) for t in zip(*vals)], index=X.index, dtype=object).astype(object)
      X = X.drop(columns=cols)
  return X


def engineer(X_train, y_train, X_test):
  return _feat(X_train), _feat(X_test)


def postprocess(pred, y_orig, X_test):
  return pred
