import numpy as np

TABFM_KWARGS = {}


def folds(X_train, y_train):
  from sklearn.model_selection import RepeatedKFold
  rkf = RepeatedKFold(n_splits=10, n_repeats=14, random_state=0)
  return list(rkf.split(X_train))


def preprocess(X_train, y_train, X_test):
  return X_train, y_train, X_test


def _feat(d):
  d = d.copy()
  binder = d['Cement'] + d['BlastFurnaceSlag'] + d['FlyAsh']
  d['ceff'] = d['Cement'] + 1.0 * d['BlastFurnaceSlag'] + 0.4 * d['FlyAsh']
  d['binder'] = binder
  d['water_cement'] = d['Water'] / d['Cement']
  d['water_binder_vol'] = d['Water'] / (d['Cement'] / 3.15 + d['BlastFurnaceSlag'] / 2.90
                                        + d['FlyAsh'] / 2.30)
  d['log_age'] = np.log(d['Age'])
  agg = d['CoarseAggregate'] + d['FineAggregate']
  d['agg_total'] = agg
  d['fine_frac'] = d['FineAggregate'] / agg
  d['sp_binder'] = d['Superplasticizer'] / binder
  return d.drop(columns=['CoarseAggregate', 'FineAggregate', 'Superplasticizer', 'Cement',
                         'BlastFurnaceSlag'])


def engineer(X_train, y_train, X_test):
  return _feat(X_train), _feat(X_test)


def postprocess(pred, y_orig, X_test):
  return pred
