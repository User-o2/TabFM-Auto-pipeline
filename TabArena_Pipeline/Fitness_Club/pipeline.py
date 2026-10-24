import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

TABFM_KWARGS = {'norm_methods': ['quantile', 'power'], 'permute_categorical': True}

DROP = ('day_of_week', 'weight', 'days_before')   # columns TabFM does better without
PRIOR_C = 0.05                                    # inverse shrinkage of the prior's logistic regression
PRIOR_K = 5                                       # folds of the prior's out-of-fold column


def folds(X_train, y_train):
  rskf = RepeatedStratifiedKFold(n_splits=3, n_repeats=5, random_state=0)
  return list(rskf.split(np.zeros(len(y_train)), np.asarray(y_train)))


def preprocess(X_train, y_train, X_test):
  return X_train, y_train, X_test


def engineer(X_train, y_train, X_test):
  keep = [c for c in X_train.columns if c not in DROP]
  Xtr, Xte = X_train[keep].copy(), X_test[keep].copy()
  y = np.asarray(y_train) == 'Yes'
  if len(np.unique(y)) > 1:      # `check` runs on a slice that can be single-class
    d_tr = _design(Xtr)
    d_te = _design(Xte).reindex(columns=d_tr.columns, fill_value=0.0)
    Xtr['prior'], Xte['prior'] = _oof_score(d_tr.to_numpy(float), y, d_te.to_numpy(float))
  return Xtr, Xte


def _design(X):
  """Numeric columns as they are, string columns as dummies."""
  num = X.select_dtypes(include=[np.number]).drop(columns=[c for c in ('row_id',) if c in X.columns])
  obj = [c for c in X.columns if c not in num.columns and c != 'row_id']
  num = num.astype(float)
  return pd.concat([num.fillna(num.median())] +
                   [pd.get_dummies(X[c].astype(str), prefix=c, dtype=float) for c in obj], axis=1)


def _oof_score(A, y, B):
  """Out-of-fold decision function on the training rows, full-fit one on the test rows."""
  from sklearn.linear_model import LogisticRegression
  from sklearn.model_selection import StratifiedKFold
  from sklearn.pipeline import make_pipeline
  from sklearn.preprocessing import StandardScaler
  mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=PRIOR_C, max_iter=5000))
  oof = np.zeros(len(A))
  for tr, va in StratifiedKFold(PRIOR_K, shuffle=True, random_state=0).split(A, y):
    oof[va] = mk().fit(A[tr], y[tr]).decision_function(A[va])
  return oof, mk().fit(A, y).decision_function(B)


def postprocess(pred, y_orig, X_test):
  return pred
