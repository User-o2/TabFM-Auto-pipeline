import numpy as np
import pandas as pd

TABFM_KWARGS = {'average_logits': False, 'class_shift': False,
                'n_feature_crosses': 'sqrt', 'n_svd_features': 'sqrt',
                'norm_methods': ['power'], 'enable_nnls': True, 'nnls_beta': 1.0}

N_REPEATS = 12
ZERO_IS_MISSING = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']


def folds(X_train, y_train):
  from sklearn.model_selection import RepeatedStratifiedKFold
  rskf = RepeatedStratifiedKFold(n_splits=3, n_repeats=N_REPEATS, random_state=0)
  return list(rskf.split(X_train, y_train))


def _demissing(X):
  X = X.copy()
  for c in ZERO_IS_MISSING:
    X[c] = X[c].astype(float).replace(0.0, np.nan)
  return X


DROP_FRAC = 0.10


def _clean(X, y):
  """Drop the rows whose own label is least consistent with their measurements."""
  from sklearn.linear_model import LogisticRegression
  from sklearn.model_selection import StratifiedKFold
  from sklearn.pipeline import make_pipeline
  from sklearn.impute import SimpleImputer
  from sklearn.preprocessing import StandardScaler
  A = _ratios(X).drop(columns=['row_id'], errors='ignore').to_numpy(float)
  A[~np.isfinite(A)] = np.nan
  t = (y.to_numpy() == np.unique(y.to_numpy())[1]).astype(int)
  p = np.zeros(len(A))
  for tri, vai in StratifiedKFold(10, shuffle=True, random_state=11).split(A, t):
    m = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                      LogisticRegression(C=1.0, max_iter=3000))
    m.fit(A[tri], t[tri])
    p[vai] = m.predict_proba(A[vai])[:, 1]
  agree = np.where(t == 1, p, 1.0 - p)
  keep = np.sort(np.argsort(agree)[int(round(DROP_FRAC * len(A))):])
  return X.iloc[keep].reset_index(drop=True), y.iloc[keep].reset_index(drop=True)


def preprocess(X_train, y_train, X_test):
  X_train, y_train = _clean(_demissing(X_train), y_train)
  return X_train, y_train, _demissing(X_test)


def _ratios(X):
  X = X.copy()
  X['Age_cap50'] = X['Age'].clip(upper=50)
  X['Age_cap40'] = X['Age'].clip(upper=40)
  X['BP_per_Age'] = X['BloodPressure'] / X['Age']
  X['Preg_per_Age'] = X['Pregnancies'] / X['Age']
  X['Age_per_BP'] = X['Age'] / X['BloodPressure']
  X['Skin_per_DPF'] = X['SkinThickness'] / X['DiabetesPedigreeFunction']
  X['Ins_per_Age'] = X['Insulin'] / X['Age']
  X['sqrt_Preg'] = np.sqrt(X['Pregnancies'])
  X['log_DPF'] = np.log(X['DiabetesPedigreeFunction'])
  X['log_Glucose'] = np.log(X['Glucose'])
  X['sqrt_Insulin'] = np.sqrt(X['Insulin'])
  return X.replace([np.inf, -np.inf], np.nan)


def engineer(X_train, y_train, X_test):
  X_train, X_test = _ratios(X_train), _ratios(X_test)
  return X_train, X_test


def postprocess(pred, y_orig, X_test):
  return pred
