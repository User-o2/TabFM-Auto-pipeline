import numpy as np

TABFM_KWARGS = {'average_logits': True, 'softmax_temperature': 1.0, 'norm_methods': ['quantile']}
DRAWS, FRAC = 5, 0.95
DROP = ['cam_col', 'fiber_ID']
PAIRS = [('u', 'r'), ('g', 'r'), ('r', 'i'), ('r', 'z')]
ANCHOR_BANDS = ['g', 'r', 'i']


def preprocess(X_train, y_train, X_test):
  return X_train, y_train, X_test


def _basis(df):
  out = df.drop(columns=DROP).copy()
  for a, b in PAIRS:
    out[f'{a}_{b}'] = out[a] - out[b]
  out['mag'] = out[ANCHOR_BANDS].mean(axis=1)
  return out.drop(columns=['u', 'g', 'r', 'i', 'z'])


def engineer(X_train, y_train, X_test):
  return _basis(X_train), _basis(X_test)


def sample(X_train, y_train, X_test, max_rows):
  n = len(X_train)
  rng = np.random.default_rng(0)
  k = int(round(FRAC * n))
  out = [np.arange(n)]
  while len(out) < DRAWS:
    out.append(np.sort(rng.choice(n, k, replace=False)))
  return out


def postprocess(pred, y_orig, X_test):
  return pred
