import os
import numpy as np
import pandas as pd

TABFM_KWARGS = {"n_feature_crosses": "sqrt"}

N_FOLDS = 5           # own CV: 2000 context rows per fold, closer to the 2500 the final fits
TOP_K = 220           # raw descriptors kept
N_SVD = 40            # SVD components of the full matrix appended
DECORR = 0.9          # skip a candidate correlated above this with one already picked
ET_SEEDS = 2          # ExtraTrees seeds the importance is averaged over
RF_TREES = 500        # auxiliary RandomForest, blended in postprocess
RF_CRIT = "entropy"   # best single auxiliary offline: 0.1398 vs 0.1410 for gini
RF_WEIGHT = 0.05      # weight of the RF ranking; TabFM keeps 0.95
SIDE = ".side_rf.npy"
DUMP = "oof_dump.csv"
TAG = "final"


def folds(X_train, y_train):
  from sklearn.model_selection import StratifiedKFold
  y = np.asarray(y_train).ravel().astype(str)
  return list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=0).split(X_train, y))


def preprocess(X_train, y_train, X_test):
  return X_train, y_train, X_test


def _plain(df):
  # pandas may hand back an arrow-backed column Index, which the judge cannot unpickle
  df.columns = pd.Index([str(c) for c in df.columns], dtype=object)
  return df


def _codes(y_train):
  return pd.factorize(np.asarray(y_train).ravel().astype(str), sort=True)[0]


def _usable_cols(X_train):
  """Every descriptor except the id column, constants and exact duplicates on this fold."""
  cols = [c for c in X_train.columns if c != "row_id"]
  A = X_train[cols].to_numpy(dtype=np.float64)
  keep, seen = [], set()
  for j, c in enumerate(cols):
    v = A[:, j]
    if v.min() == v.max():
      continue
    k = v.tobytes()
    if k in seen:
      continue
    seen.add(k)
    keep.append(c)
  return keep


def _importance(A, y):
  from sklearn.ensemble import ExtraTreesClassifier
  imp = np.zeros(A.shape[1])
  for seed in range(ET_SEEDS):
    et = ExtraTreesClassifier(n_estimators=400, random_state=seed, n_jobs=-1,
                              max_features="sqrt", min_samples_leaf=2)
    et.fit(A, y)
    imp += et.feature_importances_
  return imp


def _select(A, imp, k):
  """Top-k by importance, skipping anything too correlated with an already-picked column."""
  cand = np.argsort(-imp)[:min(A.shape[1], 4 * k)]
  Z = A[:, cand].astype(np.float64)
  Z = (Z - Z.mean(0)) / np.clip(Z.std(0), 1e-9, None)
  picked = []
  for j in range(len(cand)):
    if picked and np.abs(Z[:, picked].T @ Z[:, j]).max() / len(Z) > DECORR:
      continue
    picked.append(j)
    if len(picked) == k:
      break
  return sorted(cand[picked].tolist())


def engineer(X_train, y_train, X_test):
  from sklearn.decomposition import TruncatedSVD
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.preprocessing import StandardScaler

  cols = _usable_cols(X_train)
  y = _codes(y_train)
  A = X_train[cols].to_numpy(dtype=np.float32)
  B = X_test[cols].to_numpy(dtype=np.float32)
  single = len(np.unique(y)) < 2          # only happens on `judge.py check`'s 150-row slice

  imp = np.zeros(len(cols)) if single else _importance(A, y)
  sel = [cols[i] for i in _select(A, imp, TOP_K)]
  Xtr, Xte = [X_train[sel].reset_index(drop=True)], [X_test[sel].reset_index(drop=True)]

  sc = StandardScaler().fit(np.vstack([A, B]))
  svd = TruncatedSVD(N_SVD, random_state=0).fit(np.vstack([sc.transform(A), sc.transform(B)]))
  names = [f"svd{i}" for i in range(N_SVD)]
  Xtr.append(pd.DataFrame(svd.transform(sc.transform(A)), columns=names))
  Xte.append(pd.DataFrame(svd.transform(sc.transform(B)), columns=names))

  if not single:                          # second opinion, kept out of the context, merged in postprocess
    rf = RandomForestClassifier(RF_TREES, criterion=RF_CRIT, n_jobs=-1, random_state=0)
    rf.fit(A, y)
    np.save(SIDE, rf.predict_proba(B)[:, 1].astype(np.float64))
  return _plain(pd.concat(Xtr, axis=1)), _plain(pd.concat(Xte, axis=1))


def _rank(v):
  return pd.Series(np.asarray(v, dtype=float)).rank(method="average").to_numpy() / (len(v) + 1.0)


def _dump(pred, X_test, side):
  """My own record of TabFM's fold predictions; nothing downstream reads it."""
  try:
    d = pd.DataFrame({"tag": TAG, "row_id": np.asarray(X_test["row_id"]),
                      "tabfm": np.asarray(pred[pred.columns[-1]], dtype=float)})
    d["side"] = np.asarray(side, dtype=float) if side is not None else np.nan
    d.to_csv(DUMP, mode="a", header=not os.path.exists(DUMP), index=False)
  except Exception:
    pass


def postprocess(pred, y_orig, X_test):
  side = np.load(SIDE) if os.path.exists(SIDE) else None
  if side is not None:
    os.remove(SIDE)
    if len(side) != len(pred):
      side = None
  _dump(pred, X_test, side)
  if side is None or RF_WEIGHT <= 0:
    return pred
  pos = pred.columns[-1]
  blend = (1.0 - RF_WEIGHT) * _rank(pred[pos].to_numpy()) + RF_WEIGHT * _rank(side)
  out = pred.copy()
  out[pos] = blend
  for c in out.columns:
    if c != pos:
      out[c] = 1.0 - blend
  return out
