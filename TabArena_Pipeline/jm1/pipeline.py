import numpy as np
from sklearn.model_selection import StratifiedKFold

# class_shift randomises the label permutation per member; with two classes it buys
# no diversity, and dropping it helped slightly on every fold set.
TABFM_KWARGS = {'class_shift': False}

N_FOLDS = 5
FOLD_SEED = 2

# The raw measurements, minus l/i/b/t which are exact functions of v, d and e.
BASE = ['loc', 'v(g)', 'ev(g)', 'iv(g)', 'lOCode', 'lOComment', 'lOBlank',
        'locCodeAndComment', 'uniq_Op', 'uniq_Opnd', 'total_Op', 'total_Opnd',
        'branchCount']
EPS = 1e-6


def folds(X_train, y_train):
  """Rows are independent software modules with no time or group structure, so
  plain stratified CV mirrors how the held-out split was made."""
  y = np.asarray(y_train).ravel()
  skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=FOLD_SEED)
  return list(skf.split(np.zeros(len(y)), y))


def preprocess(X_train, y_train, X_test):
  return X_train, y_train, X_test


def _shape(X):
  """Size-normalised ratios and identity residuals, signed-log1p compressed."""
  loc, vg = X['loc'], X['v(g)']
  n1, n2, N1, N2 = X['uniq_Op'], X['uniq_Opnd'], X['total_Op'], X['total_Opnd']
  eta, N = n1 + n2, N1 + N2
  raw = {
      # how branchy the code is for its size
      'cyc_dens': vg / (loc + EPS),
      'ev_ratio': X['ev(g)'] / (vg + EPS),
      'iv_ratio': X['iv(g)'] / (vg + EPS),
      'bc_ratio': X['branchCount'] / (vg + EPS),
      'bc_resid': X['branchCount'] - 2 * vg + 1,
      # what the lines are made of
      'comment_dens': X['lOComment'] / (loc + EPS),
      'blank_dens': X['lOBlank'] / (loc + EPS),
      'code_dens': X['lOCode'] / (loc + EPS),
      'loc_resid': loc - (X['lOCode'] + X['lOComment'] + X['lOBlank']),
      # how dense the Halstead terms are per line
      'n_per_loc': X['n'] / (loc + EPS),
      'v_per_loc': X['v'] / (loc + EPS),
      'e_per_loc': X['e'] / (loc + EPS),
      # vocabulary, and how often it is reused
      'eta': eta,
      'N': N,
      'N_per_eta': N / (eta + EPS),
      'op_ratio': n1 / (n2 + EPS),
      'tot_ratio': N1 / (N2 + EPS),
      'rep_op': N1 / (n1 + EPS),
      'rep_opnd': N2 / (n2 + EPS),
      # where the recorded Halstead values disagree with their own definitions
      'v_resid': X['v'] - N * np.log2(eta.replace(0, np.nan)),
      'e_resid': X['e'] - X['v'] * X['d'],
      'n_resid': X['n'] - N,
  }
  out = {}
  for k, v in raw.items():
    v = np.asarray(v, dtype=float)
    out[k] = np.sign(v) * np.log1p(np.abs(v))
  return out


def engineer(X_train, y_train, X_test):
  def build(X):
    Z = X[[c for c in X.columns if c in BASE or c == 'row_id']].copy()
    for k, v in _shape(X).items():
      Z[k] = v
    return Z
  return build(X_train), build(X_test)


def postprocess(pred, y_orig, X_test):
  return pred
