import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import QuantileTransformer

CFG = {'tf': 'slog', 'miss_ind': 1, 'n_miss': 1, 'dedup': 0.98}

TABFM_KWARGS = {'feat_shuffle_method': 'none'}


def folds(X_train, y_train):
  y = np.asarray(y_train).ravel()
  cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=12345)
  return [(tr, va) for tr, va in cv.split(np.zeros(len(y)), y)]


def preprocess(X_train, y_train, X_test):
  return X_train, y_train, X_test


def _plain(names, mats, n):
  """Float frame with an object-dtype column index: pandas 3 builds pyarrow-backed string indexes
  inside the judge's sandbox and the parent process cannot unpickle those."""
  return pd.DataFrame(np.column_stack(mats).reshape(n, len(names)),
                      columns=pd.Index([str(c) for c in names], dtype=object))


def _slog(a):
  return np.sign(a) * np.log1p(np.abs(a))


def engineer(X_train, y_train, X_test):
  cols = [c for c in X_train.columns if c != "row_id"]
  ids = [c for c in X_train.columns if c == "row_id"]
  cols0 = list(cols)
  A = X_train[cols].to_numpy(dtype=float)
  B = X_test[cols].to_numpy(dtype=float)
  names, blocks_tr, blocks_te = [], [], []
  for c in ids:
    names.append(c)
    blocks_tr.append(X_train[c].to_numpy(dtype=float)[:, None])
    blocks_te.append(X_test[c].to_numpy(dtype=float)[:, None])

  t = CFG.get("tf", "raw")
  if t in ("wins", "wins_slog"):                      # clip to train quantiles, fold-fitted
    q = CFG.get("wq", 0.01)
    lo = np.nanquantile(A, q, axis=0)
    hi = np.nanquantile(A, 1 - q, axis=0)
    A, B = np.clip(A, lo, hi), np.clip(B, lo, hi)
  if t == "mad_slog":                                 # scale by a robust spread before compressing
    med = np.nanmedian(A, axis=0)
    mad = np.nanmedian(np.abs(A - med), axis=0)
    s = np.where(mad > 0, mad * 1.4826, 1.0)
    A, B = (A - med) / s, (B - med) / s
  if t == "sslog":                                    # scale only -- keep the sign, which is meaningful here
    s = np.nanmedian(np.abs(A), axis=0)
    s = np.where(s > 0, s, 1.0)
    A, B = A / s, B / s
  if t in ("slog", "wins_slog", "mad_slog", "sslog"):
    A, B = _slog(A), _slog(B)
  elif t == "cbrt":                                   # a milder compression than the log
    A, B = np.cbrt(A), np.cbrt(B)
  elif t == "rankg":                                  # rank -> gaussian, fold-fitted
    qt = QuantileTransformer(output_distribution="normal",
                             n_quantiles=min(1000, len(A)), subsample=10**9, random_state=0)
    A, B = qt.fit_transform(A), qt.transform(B)
  f = CFG.get("fill")
  if f == "med":                                      # fold-fitted median fill (indicators keep theinformation carried by the pattern)
    m = np.nanmedian(A, axis=0)
    m = np.where(np.isfinite(m), m, 0.0)
    A, B = np.where(np.isnan(A), m, A), np.where(np.isnan(B), m, B)
  elif f == "sent":                                   # NaN as one distinct out-of-range value
    v = CFG.get("sent", -9.0)
    A, B = np.where(np.isnan(A), v, A), np.where(np.isnan(B), v, B)

  if CFG.get("dedup"):                                # drop columns that duplicate an earlier one
    keep, seen = [], []
    for j in range(A.shape[1]):
      v = A[:, j]
      ok = np.isfinite(v)
      if ok.sum() < 2 or np.nanstd(v) == 0:
        continue
      dup = False
      for k in seen:
        m = ok & np.isfinite(A[:, k])
        if m.sum() > 2 and abs(np.corrcoef(v[m], A[m, k])[0, 1]) > CFG["dedup"]:
          dup = True
          break
      if not dup:
        seen.append(j)
        keep.append(j)
    cols = [cols[j] for j in keep]
    A, B = A[:, keep], B[:, keep]

  if CFG.get("order") == "mi":        # with column shuffling off, the order TabFM sees is mine to choose
    from sklearn.feature_selection import mutual_info_classif
    yy = np.asarray(y_train).ravel()
    yy = (yy == np.unique(yy)[-1]).astype(int) if len(np.unique(yy)) > 1 else np.zeros(len(yy), int)
    med = np.nanmedian(A, axis=0)
    Z = np.where(np.isnan(A), np.where(np.isfinite(med), med, 0.0), A)
    o = np.argsort(mutual_info_classif(Z, yy, random_state=0))[::-1]
    cols = [cols[j] for j in o]
    A, B = A[:, o], B[:, o]

  if not CFG.get("drop_base", 0):
    names += list(cols)
    blocks_tr.append(A)
    blocks_te.append(B)

  if CFG.get("slog2") and t != "slog":                # a second view of the same columns
    names += ["sl_" + c for c in cols0]
    blocks_tr.append(_slog(X_train[cols0].to_numpy(dtype=float)))
    blocks_te.append(_slog(X_test[cols0].to_numpy(dtype=float)))
  NA_tr = np.isnan(X_train[cols0].to_numpy(dtype=float))
  NA_te = np.isnan(X_test[cols0].to_numpy(dtype=float))
  if CFG.get("miss_ind"):
    mc = [j for j in range(len(cols0)) if NA_tr[:, j].any()]
    names += ["mi_" + cols0[j] for j in mc]
    blocks_tr.append(NA_tr[:, mc].astype(float))
    blocks_te.append(NA_te[:, mc].astype(float))
  if CFG.get("n_miss"):
    names.append("n_missing")
    blocks_tr.append(NA_tr.sum(1, keepdims=True).astype(float))
    blocks_te.append(NA_te.sum(1, keepdims=True).astype(float))

  R_tr = X_train[cols0].to_numpy(dtype=float)
  R_te = X_test[cols0].to_numpy(dtype=float)
  if CFG.get("neg_ind"):                              # sign flags: negative equity / profit / working capital
    frac = np.nanmean(R_tr < 0, axis=0)
    nc = [j for j in range(len(cols0)) if 0.01 < frac[j] < 0.5]
    names += ["neg_" + cols0[j] for j in nc]
    blocks_tr.append((R_tr[:, nc] < 0).astype(float))
    blocks_te.append((R_te[:, nc] < 0).astype(float))

  y = np.asarray(y_train).ravel()
  yb = (y == np.unique(y)[-1]).astype(int) if len(np.unique(y)) > 1 else np.zeros(len(y), int)
  need_rank = CFG.get("pairs") or CFG.get("topk") or CFG.get("oof")
  if need_rank:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.inspection import permutation_importance  # noqa: F401  (kept for reference)
    S_tr = _slog(R_tr)
    S_te = _slog(R_te)
  if CFG.get("pairs"):                                # log-ratios of the k most informative columns
    from sklearn.feature_selection import mutual_info_classif
    Z = np.where(np.isfinite(S_tr), S_tr, np.nanmedian(np.where(np.isfinite(S_tr), S_tr, np.nan), axis=0))
    mi = mutual_info_classif(np.nan_to_num(Z), yb, random_state=0)
    top = list(np.argsort(mi)[::-1][:CFG["pairs"]])
    for a in range(len(top)):
      for b in range(a + 1, len(top)):
        i, j = top[a], top[b]
        names.append("lr_%d_%d" % (i, j))
        blocks_tr.append((S_tr[:, i] - S_tr[:, j])[:, None])
        blocks_te.append((S_te[:, i] - S_te[:, j])[:, None])
  if CFG.get("oof"):                                  # a GBDT's out-of-fold log-odds as one extra column
    from sklearn.model_selection import StratifiedKFold
    md = dict(max_iter=CFG.get("oof_iter", 200), learning_rate=0.06, max_leaf_nodes=15,
              l2_regularization=1.0, random_state=0)
    o_tr = np.zeros(len(S_tr))
    o_te = np.zeros(len(S_te))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    for a, b in skf.split(S_tr, yb):
      m = HistGradientBoostingClassifier(**md).fit(S_tr[a], yb[a])
      o_tr[b] = m.predict_proba(S_tr[b])[:, 1]
      o_te += m.predict_proba(S_te)[:, 1] / 5.0
    lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    names.append("gbdt_logodds")
    blocks_tr.append(lg(o_tr)[:, None])
    blocks_te.append(lg(o_te)[:, None])

  Xtr = _plain(names, blocks_tr, len(X_train))
  Xte = _plain(names, blocks_te, len(X_test))
  if CFG.get("topk"):                                 # keep the k columns a GBDT actually uses
    from sklearn.ensemble import HistGradientBoostingClassifier as H
    V = Xtr.to_numpy(float)
    m = H(max_iter=150, learning_rate=0.08, max_leaf_nodes=15, random_state=0).fit(V, yb)
    from sklearn.inspection import permutation_importance as PI
    imp = PI(m, V, yb, n_repeats=3, random_state=0, scoring="roc_auc", n_jobs=4).importances_mean
    keep = list(np.argsort(imp)[::-1][:CFG["topk"]])
    Xtr, Xte = Xtr.iloc[:, keep], Xte.iloc[:, keep]
  return Xtr, Xte


def sample(X_train, y_train, X_test, max_rows):
  n_draw, rep, sub = CFG.get("draws", 1), CFG.get("pos_rep", 0), CFG.get("subsample", 0)
  if n_draw <= 1 and not rep and not sub:
    return [np.arange(len(X_train))]
  y = np.asarray(y_train).ravel()
  pos = np.flatnonzero(y == np.unique(y)[-1])
  neg = np.setdiff1d(np.arange(len(y)), pos)
  if len(pos) == 0 or len(neg) == 0:
    return [np.arange(len(y))]
  rng = np.random.default_rng(0)
  out = []
  for _ in range(max(1, n_draw)):
    if sub:
      idx = np.sort(np.concatenate([rng.choice(pos, max(1, int(round(len(pos) * sub))), replace=False),
                                    rng.choice(neg, max(1, int(round(len(neg) * sub))), replace=False)]))
    else:
      idx = np.arange(len(y))
    if rep:
      idx = np.concatenate([idx] + [np.intersect1d(idx, pos)] * rep)
    out.append(idx)
  return out


def postprocess(pred, y_orig, X_test):
  return pred
