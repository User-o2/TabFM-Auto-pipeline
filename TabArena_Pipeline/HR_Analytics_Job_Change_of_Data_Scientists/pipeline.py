import numpy as np
import pandas as pd

TABFM_KWARGS = {'n_feature_crosses': 'sqrt', 'permute_categorical': True,
                'norm_methods': ['none', 'power', 'quantile', 'quantile_rtdl', 'robust']}

FEATS = {'city_te': False, 'city_cnt': False, 'miss': True, 'inter': False, 'drop_city': False, 'na_flags': False, 'na_level': False, 'size_rank': True}
DROP = []

TE_FOLDS = 5
TE_PRIOR = 20.0

_EXP = {'<1': 0.0, '>20': 21.0}
_LNJ = {'never': 0.0, '>4': 5.0}
_SIZE = {'<10': 5.0, '10/49': 30.0, '50-99': 75.0, '100-500': 300.0,
         '500-999': 750.0, '1000-4999': 3000.0, '5000-9999': 7500.0, '10000+': 15000.0}
_EDU = {'Primary School': 0.0, 'High School': 1.0, 'Graduate': 2.0, 'Masters': 3.0, 'Phd': 4.0}


def _num(s, mapping):
  """String column -> float, using `mapping` for the non-numeric levels."""
  s = s.astype('object')
  out = s.map(mapping)
  rest = pd.to_numeric(s.where(out.isna()), errors='coerce')
  return out.astype(float).fillna(rest).astype(float)


def _plain(d):
  """Strip pandas-3 StringDtype (pyarrow-backed) from the frame *and its column index*: the
  judge unpickles the hook output in a process that cannot always import pyarrow."""
  for c in d.columns:
    if str(d[c].dtype) == 'str':
      d[c] = d[c].astype(object)
  d.columns = pd.Index(list(map(str, d.columns)), dtype=object)
  return d


def preprocess(X_train, y_train, X_test):
  return X_train, y_train.astype(object), X_test


def _feat(X):
  d = pd.DataFrame(index=X.index)
  if 'row_id' in X.columns:                       # absent under `judge.py check`
    d['row_id'] = X['row_id']
  d['cdi'] = X['city_development_index'].astype(float)
  d['city'] = X['city'].astype('object')
  d['gender'] = X['gender'].astype('object')
  d['rel_exp'] = (X['relevent_experience'].astype('object') == 'Has relevent experience').astype(float)
  d['enrolled'] = X['enrolled_university'].astype('object')
  d['edu'] = _num(X['education_level'], _EDU)
  d['major'] = X['major_discipline'].astype('object')
  d['experience'] = _num(X['experience'], _EXP)
  d['comp_size'] = np.log10(_num(X['company_size'], _SIZE))
  d['comp_type'] = X['company_type'].astype('object')
  d['last_new_job'] = _num(X['last_new_job'], _LNJ)
  d['train_hours'] = np.log1p(X['training_hours'].astype(float))
  return d


def _target_encode(key_tr, key_te, y, folds=TE_FOLDS, prior=TE_PRIOR, seed=7):
  """Smoothed mean encoding: out-of-fold for the context rows, full-fit for the query rows.

  The context rows are what TabFM attends over, so encoding them with their own label would put
  the answer in the context; the inner K-fold keeps each context row's code label-free."""
  from sklearn.model_selection import StratifiedKFold
  y = np.asarray(y, dtype=float)
  k_tr = pd.Series(np.asarray(key_tr, dtype=object).astype(str))
  k_te = pd.Series(np.asarray(key_te, dtype=object).astype(str))
  gm = float(y.mean())
  enc_tr = np.full(len(k_tr), gm)
  for i, j in StratifiedKFold(folds, shuffle=True, random_state=seed).split(k_tr, y):
    s = pd.Series(y[i]).groupby(k_tr.iloc[i].to_numpy()).agg(['sum', 'count'])
    m = (s['sum'] + prior * gm) / (s['count'] + prior)
    enc_tr[j] = k_tr.iloc[j].map(m).astype(float).fillna(gm).to_numpy()
  s = pd.Series(y).groupby(k_tr.to_numpy()).agg(['sum', 'count'])
  m = (s['sum'] + prior * gm) / (s['count'] + prior)
  return enc_tr, k_te.map(m).astype(float).fillna(gm).to_numpy()


def engineer(X_train, y_train, X_test):
  A, B = _feat(X_train), _feat(X_test)
  yb = (np.asarray(y_train, dtype=object).ravel() == 'Yes').astype(float)

  if FEATS['city_te']:
    A['city_te'], B['city_te'] = _target_encode(X_train['city'], X_test['city'], yb)

  if FEATS['city_cnt']:
    vc = pd.Series(np.asarray(X_train['city'], dtype=object).astype(str)).value_counts()
    for D, S in ((A, X_train), (B, X_test)):
      D['city_n'] = np.log1p(pd.Series(np.asarray(S['city'], dtype=object).astype(str)).map(vc).fillna(0).to_numpy())

  if FEATS['miss']:
    cols = ['gender', 'enrolled_university', 'education_level', 'major_discipline',
            'company_size', 'company_type', 'last_new_job', 'experience']
    for D, S in ((A, X_train), (B, X_test)):
      D['n_missing'] = S[cols].isna().sum(axis=1).to_numpy().astype(float)
      D['no_company'] = (S['company_size'].isna() & S['company_type'].isna()).to_numpy().astype(float)

  if FEATS['inter']:
    for D in (A, B):
      D['exp_minus_lnj'] = D['experience'] - D['last_new_job']
      D['cdi_x_exp'] = D['cdi'] * D['experience']
      D['hours_per_exp'] = D['train_hours'] / (1.0 + D['experience'])

  if FEATS['na_flags']:                          # per-column missingness, not just its count
    for c in ['gender', 'enrolled_university', 'education_level', 'major_discipline',
              'company_size', 'company_type', 'last_new_job', 'experience']:
      A['na_' + c[:6]] = X_train[c].isna().to_numpy().astype(float)
      B['na_' + c[:6]] = X_test[c].isna().to_numpy().astype(float)

  if FEATS['na_level']:                          # missing as its own category rather than NaN
    for D in (A, B):
      for c in ['city', 'gender', 'enrolled', 'major', 'comp_type']:
        D[c] = pd.Series(D[c]).where(pd.notna(D[c]), 'NA').astype(object)

  if FEATS['size_rank']:                         # equal-spaced bucket index instead of log10 headcount
    order = {k: float(i) for i, k in enumerate(
        ['<10', '10/49', '50-99', '100-500', '500-999', '1000-4999', '5000-9999', '10000+'])}
    A['comp_size'] = _num(X_train['company_size'], order)
    B['comp_size'] = _num(X_test['company_size'], order)

  if FEATS['drop_city']:
    A, B = A.drop(columns=['city']), B.drop(columns=['city'])
  if DROP:
    A, B = A.drop(columns=DROP), B.drop(columns=DROP)
  return _plain(A), _plain(B)


def postprocess(pred, y_orig, X_test):
  return pred
