import numpy as np

TABFM_KWARGS = {
    "max_num_rows": 131072,
    "norm_methods": ["none", "power", "quantile_rtdl", "robust"],
    "outlier_threshold": 10.0,
    "permute_categorical": True,
    "n_svd_features": "sqrt",
}


def preprocess(X_train, y_train, X_test):
  return X_train, y_train, X_test


def engineer(X_train, y_train, X_test):
  X_train, X_test = X_train.copy(), X_test.copy()
  for df in (X_train, X_test):
    a = df["ArrivalDelayinMinutes"].to_numpy(dtype="float64")
    d = df["DepartureDelayinMinutes"].to_numpy(dtype="float64")
    df["ArrivalDelayinMinutes"] = np.where(np.isnan(a), d, a)
  return X_train, X_test


def sample(X_train, y_train, X_test, max_rows):
  idx = np.arange(len(X_train))
  return [idx, idx[::-1].copy()]


def postprocess(pred, y_orig, X_test):
  return pred
