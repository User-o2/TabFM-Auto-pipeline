import hashlib
import pathlib

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

CONTEXT_FRACTION = 0.75
TABFM_KWARGS = {'outlier_threshold': 1000.0, 'average_logits': False}
BLEND_W = 0.60           # weight on TabFM's ranking; 1.0 is TabFM alone
PSEUDO_P = 1.0           # partner agreement needed before a test row joins the context (1.0 = off)
LOG_COUNTS = False       # log1p the integer count columns

ID_COLS = ('row_id',)
DUMP = pathlib.Path('.dump')
PARTNER = pathlib.Path('.partner')
_STATE = {}


def folds(X_train, y_train):
  from sklearn.model_selection import RepeatedStratifiedKFold
  rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0)
  return list(rskf.split(X_train, y_train))


def _key(X, cols):
  return hashlib.sha1(np.ascontiguousarray(
      X[cols].to_numpy(dtype=np.float64)).tobytes()).hexdigest()[:16]


def _rank(a):
  return pd.Series(a).rank(pct=True).to_numpy()


def _ratios(df):
  """Twelve descriptor ratios for the tree members of the blend partner. Trees split one column at
  a time, so a ratio is information they cannot reach; TabFM never sees these."""
  g = lambda c: df[c].to_numpy(float)
  hv, o, n, x = (g('Num_Heavy_Atoms'), g('Num_Oxygen_Atoms'), g('Num_Nitrogen_Atoms'),
                 g('Num_Halogen_Atoms'))
  cir, don = g('Num_Circuits'), g('Num_HBond_Donors_Atoms')
  return df.assign(r_o_hv=o / (hv + 1), r_x_hv=x / (hv + 1), r_n_o=n / (o + 1),
                   r_on_x=(o + n) / (x + 1), r_don_on=don / (o + n + 1), r_cir_hv=cir / (hv + 1),
                   r_dO_o=g('Sum_dO_EStates') / (o + 1),
                   r_benz_cir=g('Num_Substituted_BenzeneC') / (cir + 1),
                   r_prim_hv=g('Num_Terminal_PrimaryC') / (hv + 1), hetero=o + n + x,
                   c_est=g('Percentage_C_Atoms') * (hv + 1),
                   r_eneg_ip=g('Mean_Sanderson_Electronegativity') / g('Mean_Ionization_Potential'))


def _partners():
  """(model, reads_ratios) -- the trees get the ratio columns, QDA is better without them."""
  from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
  from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                                RandomForestClassifier)
  from sklearn.pipeline import make_pipeline
  from sklearn.preprocessing import QuantileTransformer
  return [(RandomForestClassifier(1000, random_state=0), True),
          (ExtraTreesClassifier(1000, min_samples_leaf=2, max_features=0.4, random_state=0), True),
          (HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, max_leaf_nodes=15,
                                          l2_regularization=1.0, early_stopping=False,
                                          random_state=0), True),
          (make_pipeline(QuantileTransformer(output_distribution='normal', n_quantiles=250,
                                             random_state=0),
                         QuadraticDiscriminantAnalysis(reg_param=0.5)), False)]


def preprocess(X_train, y_train, X_test):
  """Fits the two partner models on the real training rows -- their test scores are both the blend
  ingredient postprocess needs and the pseudo-label source -- then extends the context with the test
  rows they agree on. preprocess and engineer share a process, so _STATE carries the scores over."""
  feats = [c for c in X_train.columns if c not in ID_COLS]
  y = (np.asarray(y_train) == 'Yes').astype(int)
  if len(np.unique(y)) < 2:            # `judge.py check` hands over a single-class slice
    return X_train, y_train, X_test
  Atr, Ate = _ratios(X_train[feats]), _ratios(X_test[feats])
  with threadpool_limits(limits=1):    # ~100 cores against a 561x41 table is all OpenMP overhead
    p = np.array([m.fit(*( (Atr, y) if r else (X_train[feats], y) ))
                  .predict_proba(Ate if r else X_test[feats])[:, 1] for m, r in _partners()])
  _STATE['partner'], _STATE['key'] = p, _key(X_test, feats)   # key on the untransformed test rows,
  if LOG_COUNTS:                                              # which is what postprocess is handed
    counts = [c for c in feats if X_train[c].dtype.kind == 'i' and X_train[c].max() > 3]
    X_train = X_train.assign(**{c: np.log1p(X_train[c]) for c in counts})
    X_test = X_test.assign(**{c: np.log1p(X_test[c]) for c in counts})
  if PSEUDO_P >= 1.0:
    return X_train, y_train, X_test
  hi, lo = (p > PSEUDO_P).all(axis=0), (p < 1 - PSEUDO_P).all(axis=0)
  take = np.where(hi | lo)[0]
  if not len(take):
    return X_train, y_train, X_test
  lab = pd.Series(np.where(hi[take], 'Yes', 'No')).astype(y_train.dtype)
  return (pd.concat([X_train, X_test.iloc[take]], ignore_index=True),
          pd.concat([y_train, lab], ignore_index=True), X_test)


def engineer(X_train, y_train, X_test):
  TABFM_KWARGS['max_num_rows'] = int(round(CONTEXT_FRACTION * len(X_train)))
  try:
    if 'partner' in _STATE:
      PARTNER.mkdir(exist_ok=True)
      np.save(PARTNER / f"{_STATE['key']}.npy", _STATE['partner'])
  except Exception:
    pass          # postprocess finds no file and returns TabFM's own probabilities unblended
  return X_train, X_test


def postprocess(pred, y_orig, X_test):
  feats = [c for c in X_test.columns if c not in ID_COLS]
  key = _key(X_test, feats)
  f = PARTNER / f'{key}.npy'
  part = np.load(f) if f.exists() else None
  try:                                 # the offline harness joins these back to data/train.csv
    DUMP.mkdir(exist_ok=True)
    np.savez(DUMP / f'{key}.npz', X=np.ascontiguousarray(X_test[feats].to_numpy(dtype=np.float64)),
             cols=np.array(feats), pred=pred.to_numpy(dtype=np.float64),
             classes=np.array([str(c) for c in pred.columns]),
             partner=np.zeros((0, len(pred))) if part is None else part)
  except Exception:
    pass
  if 'Yes' not in pred.columns or part is None:
    return pred                        # `judge.py check`, or a probe run with no partner fitted
  yes = pred['Yes'].to_numpy(dtype=np.float64)
  s = BLEND_W * _rank(yes) + (1 - BLEND_W) * np.mean([_rank(p) for p in part], axis=0)
  out = pred.copy()   # not a fresh DataFrame: a new string column index does not survive the
  out['Yes'] = s      # hook's pickle round-trip on this pandas build (pyarrow-backed str dtype)
  out['No'] = 1.0 - s
  return out
