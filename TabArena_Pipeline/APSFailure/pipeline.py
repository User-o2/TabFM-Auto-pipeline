import numpy as np
import pandas as pd

TABFM_KWARGS = {"n_svd_features": "sqrt"}

HIST_GROUPS = ["ag", "ay", "az", "ba", "cn", "cs", "ee"]



def preprocess(X_train, y_train, X_test):
  return X_train, y_train, X_test


def _build(X):
  out = X.copy()
  bins_all = []
  tots = {}
  for g in HIST_GROUPS:
    cols = [f"{g}_{i:03d}" for i in range(10)]
    cols = [c for c in cols if c in X.columns]
    if len(cols) < 2:
      continue
    B = X[cols].to_numpy(dtype=float)
    tot = np.nansum(B, axis=1)
    allna = np.isnan(B).all(axis=1)
    tot = np.where(allna, np.nan, tot)
    denom = np.where(tot > 0, tot, np.nan)
    F = B / denom[:, None]
    idx = np.arange(len(cols), dtype=float)
    com = np.nansum(F * idx[None, :], axis=1)
    com = np.where(np.isnan(denom), np.nan, com)
    tots[f"{g}_tot"] = np.log1p(tot)
    out[f"{g}_com"] = com
    for j, c in enumerate(cols):
      out[f"{g}_f{j}"] = F[:, j]
    bins_all += cols

  out = out.drop(columns=bins_all)
  raw = [c for c in X.columns if c not in bins_all and c != "row_id"]
  out[raw] = np.log1p(out[raw].to_numpy(dtype=float))
  for k, v in tots.items():
    out[k] = v
  out["n_na"] = X.isna().sum(axis=1).to_numpy()
  return out


def engineer(X_train, y_train, X_test):
  return _build(X_train), _build(X_test)


def postprocess(pred, y_orig, X_test):
  return pred
