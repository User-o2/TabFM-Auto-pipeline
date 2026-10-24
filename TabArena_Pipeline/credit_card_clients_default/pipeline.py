import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

TABFM_KWARGS = {'n_feature_crosses': 'sqrt', 'norm_methods': ['none', 'power', 'quantile', 'robust', 'quantile_rtdl'], 'class_shift': False, 'outlier_threshold': 3.0, 'average_logits': False}

PAY = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
BILL = [f'BILL_AMT{i}' for i in range(1, 7)]
AMT = [f'PAY_AMT{i}' for i in range(1, 7)]


def folds(X_train, y_train):
  y = np.asarray(y_train).ravel()
  skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
  return list(skf.split(np.zeros(len(y)), y))


def preprocess(X_train, y_train, X_test):
  return X_train, y_train, X_test


def postprocess(pred, y_orig, X_test):
  return pred


def engineer(X_train, y_train, X_test):
  def f(X):
    X = X.copy()
    B = X[BILL].values.astype(np.float64)
    A = X[AMT].values.astype(np.float64)
    L = X['LIMIT_BAL'].values.astype(np.float64)
    U = B / L[:, None]
    X[BILL[0]] = U[:, 0]
    X[BILL[5]] = U[:, 5]
    for i in range(1, 5):
      X[BILL[i]] = U[:, i - 1] - U[:, i]
    for i, c in enumerate(AMT):
      denom = np.maximum(B[:, i + 1], 1.0) if i < 5 else L
      X[c] = np.clip(A[:, i] / denom, 0.0, 1.75)
    e = X['EDUCATION'].values.astype(np.int64).copy()
    e[(e == 0) | (e >= 4)] = 4
    X['EDUCATION'] = e
    m = X['MARRIAGE'].values.astype(np.int64).copy()
    m[m == 0] = 3
    X['MARRIAGE'] = m
    return X
  return f(X_train), f(X_test)
