import numpy as np
import pandas as pd

TABFM_KWARGS = {'n_feature_crosses': 'sqrt', 'outlier_threshold': 2.0, 'feat_shuffle_method': 'none'}

F = ['CIC0', 'SM1_Dz(Z)', 'GATS1i', 'NdsCH', 'NdssC', 'MLOGP']
NEW = [('r_mg', 'm / g'), ('r_sg', 's / g'), ('r_cg', 'c / g'), ('p_mc', 'm * c'), ('p_cs', 'c * s'), ('p_ms', 'm * s')]
STATE = '.pipe_state.pkl'   # engineer and postprocess run in separate processes; the workspace is the only channel


def folds(X_train, y_train):
  """Repeated stratified CV: 3 shuffles x 5 folds, so every row is scored three times under
  three different training sets. The mean is the selection signal; a single 5-fold split moves
  by ~0.002 on nothing but which rows landed together."""
  from sklearn.model_selection import RepeatedStratifiedKFold
  y = np.asarray(y_train, dtype=float).ravel()
  bins = np.digitize(y, np.quantile(y, np.linspace(0, 1, 11)[1:-1]))
  rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=7)
  return list(rskf.split(np.zeros(len(y)), bins))


def _knn_oof(X, y, k=10):
  """Out-of-fold distance-weighted kNN estimate of the target, fitted inside the training rows only."""
  from sklearn.model_selection import KFold
  from sklearn.neighbors import KNeighborsRegressor
  from sklearn.preprocessing import StandardScaler
  A = StandardScaler().fit_transform(X[F])
  oof = np.zeros(len(A))
  for tr, va in KFold(10, shuffle=True, random_state=0).split(A):
    oof[va] = KNeighborsRegressor(k, weights='distance').fit(A[tr], y[tr]).predict(A[va])
  return oof


def preprocess(X_train, y_train, X_test):
  from sklearn.model_selection import KFold
  from sklearn.preprocessing import StandardScaler
  from sklearn.neighbors import KNeighborsRegressor
  from sklearn.svm import SVR
  from sklearn.ensemble import ExtraTreesRegressor
  y = np.asarray(y_train, dtype=float).ravel()
  A = StandardScaler().fit_transform(X_train[F])
  oof = np.zeros(len(A))
  for tr, va in KFold(10, shuffle=True, random_state=0).split(A):
    oof[va] = (KNeighborsRegressor(5, weights='distance').fit(A[tr], y[tr]).predict(A[va])
               + SVR(C=3).fit(A[tr], y[tr]).predict(A[va])
               + ExtraTreesRegressor(200, random_state=0).fit(A[tr], y[tr]).predict(A[va])) / 3
  return X_train, pd.Series((1 - 0.30) * y + 0.30 * oof, index=X_train.index), X_test


def _add(X):
  X = X.copy()
  c, s, g, m = X['CIC0'], X['SM1_Dz(Z)'], X['GATS1i'], X['MLOGP']
  h, j = X['NdsCH'], X['NdssC']
  for k, expr in NEW:
    X[k] = eval(expr)
  return X


def engineer(X_train, y_train, X_test):
  Xtr, Xte = _add(X_train), _add(X_test)

  pd.to_pickle({'X': X_train[F].reset_index(drop=True),
                'y': np.asarray(y_train, dtype=float).ravel()}, STATE)
  return Xtr, Xte


def postprocess(pred, y_orig, X_test):
  from sklearn.preprocessing import StandardScaler
  from sklearn.neighbors import KNeighborsRegressor
  st = pd.read_pickle(STATE)
  sc = StandardScaler().fit(st['X'])
  p = KNeighborsRegressor(10, weights='distance').fit(sc.transform(st['X']), st['y']).predict(sc.transform(X_test[F]))
  return (1 - 0.26) * pred + 0.26 * pd.Series(p, index=pred.index)
