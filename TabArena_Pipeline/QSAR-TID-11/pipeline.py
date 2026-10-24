import numpy as np
import pandas as pd
from sklearn.svm import SVR

TABFM_KWARGS = {'n_svd_features': 'sqrt', 'enable_nnls': True}

N_BITS = 500
W_TABFM = 0.66
W_RARE = 0.10
SVR_C = 10.0
SVR_EPS = 0.0
TVERSKY = 0.7
BLEND_FILE = '.blend.csv'


def _bits(X):
  return [c for c in X.columns if c.startswith('FCFP6_')]


def _sim(A, B, alpha=None):
  """Tanimoto (alpha=None) or symmetric Tversky: how harshly the bits only one molecule has are counted."""
  inter = A @ B.T
  na = A.sum(1)[:, None]
  nb = B.sum(1)[None, :]
  if alpha is None:
    return inter / (na + nb - inter + 1e-9)
  return inter / (inter + alpha * (na - inter) + alpha * (nb - inter) + 1e-9)


def _mean_kernel(A, B):
  return (0.5 * (_sim(A, B) + _sim(A, B, TVERSKY))).astype(np.float64)


def _kernel_pred(A, B, y):
  m = SVR(C=SVR_C, kernel='precomputed', epsilon=SVR_EPS, tol=1e-3,
          cache_size=1000).fit(_mean_kernel(A, A), y)
  return m.predict(_mean_kernel(B, A))


def preprocess(X_train, y_train, X_test):
  return X_train, y_train, X_test


def engineer(X_train, y_train, X_test):
  cols = _bits(X_train)
  A = X_train[cols].to_numpy(np.float32)
  B = X_test[cols].to_numpy(np.float32)
  y = np.asarray(y_train, dtype=np.float64).ravel()

  order = np.argsort(-A.var(0), kind='stable')
  sel = np.sort(order[:N_BITS])
  rest = np.sort(order[N_BITS:])

  try:
    base = _kernel_pred(A, B, y)
    if len(rest):                       # the bits the column cap keeps away from TabFM
      base = (1.0 - W_RARE) * base + W_RARE * _kernel_pred(A[:, rest], B[:, rest], y)
    pd.DataFrame({'row_id': X_test['row_id'].to_numpy(), 'base': base}).to_csv(BLEND_FILE, index=False)
  except Exception:
    pd.DataFrame({'row_id': [], 'base': []}).to_csv(BLEND_FILE, index=False)

  keep = [cols[i] for i in sel]
  return X_train[keep], X_test[keep]


def postprocess(pred, y_orig, X_test):
  p = np.asarray(pred, dtype=np.float64).ravel()
  base = None
  try:
    b = pd.read_csv(BLEND_FILE)
    m = pd.DataFrame({'row_id': X_test['row_id'].to_numpy()}).merge(b, on='row_id', how='left')
    if len(m) == len(p) and m['base'].notna().all():
      base = m['base'].to_numpy()
  except Exception:
    base = None
  out = p if base is None else W_TABFM * p + (1.0 - W_TABFM) * base
  try:
    pd.DataFrame({'row_id': X_test['row_id'].to_numpy(), 'pred': p,
                  'base': base if base is not None else np.nan}).to_csv('preds_dump.csv', mode='a',
                                                                        header=False, index=False)
  except Exception:
    pass
  return pd.Series(out, index=pred.index if hasattr(pred, 'index') else None)
