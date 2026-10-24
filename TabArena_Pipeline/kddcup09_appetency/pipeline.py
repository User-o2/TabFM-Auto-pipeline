import numpy as np
import pandas as pd
from scipy.stats import rankdata

TABFM_KWARGS = {'norm_methods': ['none', 'power', 'quantile', 'robust']}

KEEP_STR_MAX = 30      # raw string kept for columns with at most this many training levels
TE_FOLDS = 5
TE_SMOOTH = 60.0
N_KEEP = 200           # columns kept, ranked by in-fold univariate AUC
MIN_LEVEL = 10         # a level rarer than this is encoded as the prior on both sides


def _as_str(s):
  """plain object-dtype strings: an arrow-backed str column does not survive the hook pickle."""
  v = s.astype(object).to_numpy(dtype=object).copy()
  v[pd.isna(v)] = "__NA__"
  return pd.Series(v, dtype=object, index=s.index)


def preprocess(X_train, y_train, X_test):
  # an arrow-backed str dtype does not survive the hook pickle on every judge server
  if not isinstance(y_train, pd.DataFrame) and y_train.dtype.kind not in "fiuO":
    y_train = y_train.astype(object)
  return X_train, y_train, X_test


def _codes(st, sv):
  """integer codes for a categorical column; query values unseen in training get -1."""
  ct, uniq = pd.factorize(st, use_na_sentinel=False)
  lut = pd.Series(np.arange(len(uniq)), index=uniq)
  cv = sv.map(lut).to_numpy()
  cv = np.where(pd.isna(cv), -1, cv).astype(np.int64)
  return ct.astype(np.int64), cv, len(uniq)


def _cat_features(ct, cv, n, yb, rng):
  """count + cross-fitted target encoding for one categorical column."""
  cnt = np.bincount(ct, minlength=n).astype(np.float64)
  pos = np.bincount(ct, weights=yb, minlength=n)
  prior = yb.mean()

  known = cnt >= MIN_LEVEL

  # cross-fitted on the context rows, full-fit map on the query rows
  te_t, cnt_t = np.empty(len(ct)), np.empty(len(ct))
  order = rng.permutation(len(ct))
  scale = TE_FOLDS / (TE_FOLDS - 1.0)
  for p in np.array_split(order, TE_FOLDS):
    c_out = np.bincount(ct[p], minlength=n).astype(np.float64)
    p_out = np.bincount(ct[p], weights=yb[p], minlength=n)
    c_in, p_in = cnt - c_out, pos - p_out
    m = np.where(known, (p_in + prior * TE_SMOOTH) / (c_in + TE_SMOOTH), prior)
    te_t[p] = m[ct[p]]
    cnt_t[p] = c_in[ct[p]] * scale
  m_full = np.where(known, (pos + prior * TE_SMOOTH) / (cnt + TE_SMOOTH), prior)
  te_v = np.where(cv >= 0, m_full[np.clip(cv, 0, None)], prior)
  cnt_v = np.where(cv >= 0, cnt[np.clip(cv, 0, None)], 0.0)
  return cnt_t, cnt_v, te_t, te_v


def _pair_codes(at, av, bt, bv, nb):
  """codes of the (a, b) pair, renumbered densely; a query row with either side unseen gets -1."""
  kt, uniq = pd.factorize(at * nb + bt, use_na_sentinel=False)
  lut = pd.Series(np.arange(len(uniq)), index=uniq)
  raw = np.where((av < 0) | (bv < 0), -1, av * nb + bv)
  kv = pd.Series(raw).map(lut).to_numpy()
  kv = np.where(pd.isna(kv), -1, kv).astype(np.int64)
  return kt.astype(np.int64), kv, len(uniq)


def _auc(v, yb, npos, nneg):
  r = rankdata(v)
  return (r[yb > 0].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def _uni_score(v, yb, npos, nneg):
  """|AUC-0.5| of one column, taking the better of ranking NaN lowest / highest, so that a column
  whose signal lives in its missingness pattern scores as high as one whose signal is in its values."""
  if v.dtype.kind not in "fiu":
    return 0.0
  m = np.isnan(v)
  if not m.any():
    return abs(_auc(v, yb, npos, nneg) - 0.5)
  lo = np.where(m, -np.inf, v)
  hi = np.where(m, np.inf, v)
  return max(abs(_auc(lo, yb, npos, nneg) - 0.5), abs(_auc(hi, yb, npos, nneg) - 0.5))


def _select(ft, cat, yb):
  """Top-N columns by in-fold univariate AUC; a raw string column inherits its target encoding's score."""
  npos, nneg = float(yb.sum()), float(len(yb) - yb.sum())
  score = {}
  for c, v in ft.items():
    score[c] = _uni_score(np.asarray(v), yb, npos, nneg)
  for c in cat:
    if c in score:
      score[c] = max(score.get(c + "_te", 0.0), score.get(c + "_cnt", 0.0))
  order = sorted(score, key=lambda c: -score[c])
  return set(order[:N_KEEP])


def engineer(X_train, y_train, X_test):
  y = y_train.iloc[:, 0] if isinstance(y_train, pd.DataFrame) else y_train
  yb = (y.astype(str).to_numpy() == "1").astype(np.float64)
  rng = np.random.default_rng(0)

  cols = [c for c in X_train.columns if c != "row_id"]
  num = [c for c in cols if pd.api.types.is_numeric_dtype(X_train[c])]
  cat = [c for c in cols if c not in num]

  ft, fv = {}, {}
  for c in num:
    v = X_train[c]
    if v.notna().sum() == 0 or v.nunique(dropna=True) <= 1:
      continue
    ft[c] = v.to_numpy(dtype=np.float64)
    fv[c] = X_test[c].to_numpy(dtype=np.float64)

  codes = {}
  for c in cat:
    st, sv = _as_str(X_train[c]), _as_str(X_test[c])
    ct, cv, nu = _codes(st, sv)
    if nu <= 1:
      continue
    codes[c] = (ct, cv, nu)
    a, b, d, e = _cat_features(ct, cv, nu, yb, rng)
    ft[c + "_cnt"], fv[c + "_cnt"] = a, b
    ft[c + "_te"], fv[c + "_te"] = d, e
    if nu <= KEEP_STR_MAX:
      ft[c] = st.to_numpy()
      fv[c] = sv.to_numpy()

  keep = _select(ft, cat, yb)
  ft = {c: v for c, v in ft.items() if c in keep}
  fv = {c: v for c, v in fv.items() if c in keep}

  Xt = pd.DataFrame(ft, index=X_train.index)
  Xv = pd.DataFrame(fv, index=X_test.index)
  for c in Xt.columns:
    if Xt[c].dtype.kind not in "fiu":
      Xt[c] = Xt[c].astype(object)
      Xv[c] = Xv[c].astype(object)
  if "row_id" in X_train.columns:
    Xt["row_id"] = X_train["row_id"].to_numpy()
    Xv["row_id"] = X_test["row_id"].to_numpy()
  Xt.columns = pd.Index(list(Xt.columns), dtype=object)
  Xv.columns = pd.Index(list(Xv.columns), dtype=object)
  return Xt, Xv


def postprocess(pred, y_orig, X_test):
  return pred
