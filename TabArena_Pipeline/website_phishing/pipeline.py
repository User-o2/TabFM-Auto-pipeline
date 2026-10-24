import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold

# Every indicator takes one of these three levels; the level names re-use the target's class names
# as codes, and they carry a natural low/middle/high order (checked against the two other orders).
ORD = {"Phishy": -1.0, "Suspicious": 0.0, "Legitimate": 1.0}

TABFM_KWARGS = {
    "n_feature_crosses": "sqrt",              # numeric-only, hence the ordinal codes
    "average_logits": False,                  # required by both the calibrator and NNLS
    "multiclass_calibration_method": "vector",
    "calibration_lambda": 0.0,
    "enable_nnls": True,
    "nnls_beta": 0.25,
    "softmax_temperature": 0.85,
}


def folds(X_train, y_train):
  """3 x 10-fold: each fit sees 811 of the 902 rows, close to the 902 the final's folds give it,
  and the mean over 30 folds is quiet enough to read differences of ~0.001."""
  return list(RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1).split(X_train, y_train))


def preprocess(X_train, y_train, X_test):
  return X_train, y_train, X_test


def engineer(X_train, y_train, X_test):
  cols = [c for c in X_train.columns if c != "row_id"]

  def enc(X):
    out = X.copy()
    for c in cols:
      out[c] = out[c].map(ORD).astype(float)
    return out

  return enc(X_train), enc(X_test)


def postprocess(pred, y_orig, X_test):
  return pred
