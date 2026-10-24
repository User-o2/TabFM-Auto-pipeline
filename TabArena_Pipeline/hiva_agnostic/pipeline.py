import numpy as np
import pandas as pd

N_SVD = 32
K_CACM = 64    # raw bits selected by chi2 on CA-vs-CM *among the actives* (0 = none)
MIN_SUPP = 1
SEL_BAG = 1
USE_POPCOUNT = 1
N_DRAWS = 24
BAL_RATIO = 0.0   # >0: inactives per active inside one context; 0: plain bagging
ACT_REP = 6       # copies of the active rows inside each draw; 3/6/12 -> 0.17184/0.17132/0.17173 balanced
RARE_BALANCE = 1  # 1: repeat each rare class to the same count, so CA and CM are balanced
DRAW_FRAC = 0.8
MU_DEV, AU_DEV = 0.02, 2.0   # bound / slope for the active-vs-inactive log odds (uninformative)
MV_DEV, AV_DEV = 0.90, 50.0  # bound / slope for the CA-vs-CM log ratio -- the direction that carries signal

TABFM_KWARGS = {"max_num_rows": 600, "outlier_threshold": 2.0}


def preprocess(X_train, y_train, X_test):
  return X_train, y_train, X_test


def engineer(X_train, y_train, X_test):
  """Low-rank view of the bit matrix, fit on this fold's features only (no labels anywhere)."""
  from sklearn.decomposition import TruncatedSVD
  cols = [c for c in X_train.columns if c != "row_id"]
  A = X_train[cols].to_numpy(dtype=np.float32)
  B = X_test[cols].to_numpy(dtype=np.float32)
  k = int(min(N_SVD, A.shape[1] - 1, len(A) - 1))
  sv = TruncatedSVD(k, random_state=0).fit(np.vstack([A, B]))
  names = pd.Index(["svd_%d" % i for i in range(k)], dtype=object)   # dtype=object: a str Index
  tr = pd.DataFrame(sv.transform(A), columns=names, index=X_train.index)   # built inside the hook
  te = pd.DataFrame(sv.transform(B), columns=names, index=X_test.index)    # sandbox is pyarrow-
  if K_CACM > 0:                          # bits picked for the one direction that carries signal
    from sklearn.feature_selection import chi2
    lab = pd.Series(y_train).astype(str).to_numpy()
    major = pd.Series(lab).value_counts().index[0]
    act = lab != major
    rare = pd.Series(lab[act]).value_counts().index
    if act.sum() > 4 and len(rare) > 1:
      t = (lab[act] == rare[0]).astype(int)
      Aa = A[act]
      supp = Aa.sum(axis=0)
      ok = (supp >= MIN_SUPP) & (supp <= len(Aa) - MIN_SUPP)
      sc = np.full(A.shape[1], -np.inf)
      if ok.any() and 0 < t.sum() < len(t):
        if SEL_BAG > 1:   # average the score over bootstrap resamples: the selection is fit on ~60
          rs = np.random.default_rng(7)   # actives, and a single chi2 on that is very unstable
          acc = np.zeros(int(ok.sum()))
          for _ in range(SEL_BAG):
            i = rs.choice(len(Aa), len(Aa), replace=True)
            if 0 < t[i].sum() < len(i):
              col = Aa[i][:, ok]
              good = col.std(axis=0) > 0
              sub = np.zeros(int(ok.sum()))
              if good.any():
                sub[good] = chi2(col[:, good], t[i])[0]
              acc += sub
          sc[ok] = acc / SEL_BAG
        else:
          col = Aa[:, ok]
          good = col.std(axis=0) > 0
          sub = np.zeros(int(ok.sum()))
          if good.any():
            sub[good] = chi2(col[:, good], t)[0]
          sc[ok] = sub
      pick = np.sort(np.argsort(-sc, kind="mergesort")[:K_CACM])
      for i in pick:
        tr["b%d" % i] = A[:, i]
        te["b%d" % i] = B[:, i]
  if USE_POPCOUNT:                                                         # backed, and the judge
    tr["n_bits"] = A.sum(axis=1)                                           # host cannot unpickle it
    te["n_bits"] = B.sum(axis=1)
  return tr, te


def sample(X_train, y_train, X_test, max_rows):
  """Bagged draws: the judge averages one full TabFM pass per draw, and different row subsets
  decorrelate the members. max_num_rows also fixes each member's context at 600 rows, so a member
  sees the same amount of data in the 1708-row CV folds as in the 2563-row final folds."""
  n = len(X_train)
  rng = np.random.default_rng(12345)
  if BAL_RATIO > 0:                       # enrich the context in actives: more CA/CM pairs in view
    y = pd.Series(y_train).astype(str).to_numpy()
    major = pd.Series(y).value_counts().index[0]
    pos, neg = np.where(y != major)[0], np.where(y == major)[0]
    m = int(min(len(neg), round(BAL_RATIO * len(pos))))
    return [np.sort(np.concatenate([pos, rng.choice(neg, m, replace=False)])) for _ in range(N_DRAWS)]
  k = max(2, int(round(DRAW_FRAC * n)))
  out = [np.sort(rng.choice(n, k, replace=False)) for _ in range(N_DRAWS)]
  if ACT_REP > 1:                         # repeat the actives: the CI context is kept intact, but
    y = pd.Series(y_train).astype(str).to_numpy()          # TabFM sees each CA/CM pair more often
    vc = pd.Series(y).value_counts()
    major = vc.index[0]
    rare = [c for c in vc.index if c != major]
    top = max(int(vc[c]) for c in rare) if rare else 0
    add = []
    for c in rare:                        # RARE_BALANCE: the rarer of CA/CM gets proportionally
      r = ACT_REP - 1                     # more copies, so the two are equally represented
      if RARE_BALANCE and int(vc[c]) > 0:
        r = max(1, int(round(ACT_REP * top / int(vc[c])))) - 1
      add.append((c, r))
    out = [np.sort(np.concatenate([d] + [d[y[d] == c] for c, r in add for _ in range(r)])) for d in out]
  return out


def _dump(pred, y_orig, X_test):
  """Record TabFM's raw output so the calibration can be re-fitted offline against the OOF rows
  instead of by spending evaluations on it. No-op unless a dumps/ directory already exists."""
  import os
  import pickle
  import time
  if not os.path.isdir("dumps"):
    return
  rid = X_test["row_id"].to_numpy() if "row_id" in X_test.columns else np.arange(len(X_test))
  rec = {"pred": np.asarray(pred, dtype=float), "cols": [str(c) for c in pd.DataFrame(pred).columns],
         "row_id": np.asarray(rid),
         "prior": {str(k): int(v) for k, v in pd.Series(y_orig).astype(str).value_counts().items()}}
  with open("dumps/p_%.6f.pkl" % time.time(), "wb") as f:
    pickle.dump(rec, f)


def _match(P, pr, iters=300):
  """Scale the columns of P so its column means hit the prior pr (Saerens-style prior matching)."""
  c = np.ones(P.shape[1])
  for _ in range(iters):
    Q = P * c
    Q /= np.maximum(Q.sum(axis=1, keepdims=True), 1e-300)
    c *= (pr / np.maximum(Q.mean(axis=0), 1e-12)) ** 0.7
  Q = P * c
  return Q / np.maximum(Q.sum(axis=1, keepdims=True), 1e-300)


def _squash(x, M, A):
  return M * np.tanh(A * x / M) if M > 0 else np.zeros_like(x)


def postprocess(pred, y_orig, X_test):
  """Returns a plain ndarray in sorted-class order: the judge re-attaches the class columns, and a
  pandas 3 str/pyarrow column Index built here fails to unpickle on the judge's side."""
  try:
    _dump(pred, y_orig, X_test)
  except Exception:
    pass
  df = pd.DataFrame(pred)
  have = [str(c) for c in df.columns]
  cols = sorted(have)
  P = np.column_stack([np.asarray(df.iloc[:, have.index(c)], dtype=float) for c in cols])
  vc = pd.Series(y_orig).astype(str).value_counts()
  pr = np.array([float(vc.get(c, 0.0)) for c in cols])
  pr = np.maximum(pr / max(pr.sum(), 1e-12), 1e-9)
  pr = pr / pr.sum()
  P = _match(np.clip(P, 1e-9, 1.0), pr)

  if len(cols) != 3:                      # `check` runs on a slice that need not hold every class
    z = np.log(np.clip(P, 1e-12, 1.0)) - np.log(pr)[None, :]
    Q = pr[None, :] * np.exp(_squash(z, MV_DEV, AV_DEV))
  else:
    j = int(np.argmax(pr))                # the majority class; the other two are the rare pair
    a, b = [i for i in range(3) if i != j]
    pa, pi = P[:, a] + P[:, b], P[:, j]
    u = np.log(pa / pi) - np.log((pr[a] + pr[b]) / pr[j])                    # active vs inactive
    v = np.log(np.clip(P[:, a], 1e-12, 1.0) / np.clip(P[:, b], 1e-12, 1.0)) \
        - np.log(pr[a] / pr[b])                                              # CA vs CM
    qa = 1.0 / (1.0 + np.exp(-(np.log((pr[a] + pr[b]) / pr[j]) + _squash(u, MU_DEV, AU_DEV))))
    f = 1.0 / (1.0 + np.exp(-(np.log(pr[a] / pr[b]) + _squash(v, MV_DEV, AV_DEV))))
    Q = np.empty_like(P)
    Q[:, a], Q[:, j], Q[:, b] = qa * f, 1.0 - qa, qa * (1.0 - f)

  Q = Q / Q.sum(axis=1, keepdims=True)
  Q = np.clip(_match(Q, pr), 1e-7, 1.0)
  return Q / Q.sum(axis=1, keepdims=True)
