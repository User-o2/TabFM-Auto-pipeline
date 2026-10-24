import numpy as np
import pandas as pd

TABFM_KWARGS = {}

N_SPLITS, N_REPEATS, FOLD_SEED = 4, 3, 12345
DROP_RARE_COUNTS = 0.05        # drop numberOf* for products held by fewer than 5% of rows
EXTRA_COUNTS = ()

# --- the three string-coded ordinals, back onto the scale their sibling columns use ---
AGE = {'20-30 years': 1, '30-40 years': 2, '40-50 years': 3,
       '50-60 years': 4, '60-70 years': 5, '70-80 years': 6}
PCT = {'0%': 0, '1 - 10%': 1, '11 - 23%': 2, '24 - 36%': 3, '37 - 49%': 4,
       '50 - 62%': 5, '63 - 75%': 6, '76 - 88%': 7, '89 - 99%': 8, '100%': 9}
L4 = {'f 0': 0, 'f 1 - 49': 1, 'f 50 - 99': 2, 'f 100 - 199': 3, 'f 200 - 499': 4,
      'f 500 - 999': 5, 'f 1000 - 4999': 6, 'f 5000 - 9999': 7,
      'f 10000 - 19999': 8, 'f 20000 - ?': 9}
ORDINAL = {'avgAge': AGE, 'romanCatholic': PCT,
           'contributionPrivateThirdPartyInsurance': L4}
NOMINAL = ['customerSubtype', 'customerMainType']

# the 36 M* columns that are percentage-of-neighbourhood bands (the rest of the M* block are
# counts, ages or purchasing-power classes and are left alone), by the ladder they belong to
SOCIO_GROUPS = {
    'religion': ['romanCatholic', 'protestant', 'otherReligion', 'noReligion'],
    'marital': ['married', 'livingTogether', 'otherRelation', 'singles'],
    'children': ['householdWithoutChildren', 'householdWithChildren'],
    'education': ['highLevelEducation', 'mediumLevelEducation', 'lowerLevelEducation'],
    'occupation': ['highStatus', 'entrepreneur', 'farmer', 'middleManagement',
                   'skilledLabourers', 'unskilledLabourers'],
    'class': ['socialClassA', 'socialClassB1', 'socialClassB2', 'socialClassC', 'socialClassD'],
    'housing': ['rentedHouse', 'homeOwners'],
    'car': ['oneCar', 'twoCars', 'noCar'],
    'health': ['nationalHealthService', 'privateHealthInsurance'],
    'income': ['incomeLessThan30k', 'income30To45k', 'income45To75k', 'income75To122k',
               'incomeAbove123k'],
}
PCT_COLS = [c for g in SOCIO_GROUPS.values() for c in g]
INCOME = SOCIO_GROUPS['income']
EDU = SOCIO_GROUPS['education']


def folds(X_train, y_train):
  """12 folds: every training row is validated three times, on three different partitions."""
  from sklearn.model_selection import RepeatedStratifiedKFold
  rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=FOLD_SEED)
  return list(rskf.split(X_train, y_train))


def _decode(X):
  X = X.copy()
  for c, m in ORDINAL.items():
    if c in X.columns:
      X[c] = pd.Series(X[c], index=X.index).map(m).astype('float64')
  for c in NOMINAL:                    # arrow-backed strings do not survive the hook pickle
    if c in X.columns:
      X[c] = X[c].astype(object)
  return X


def preprocess(X_train, y_train, X_test):
  return _decode(X_train), y_train, _decode(X_test)


def _counts(X):
  """The 17 row-wise counts, in the order they are appended to the table."""
  P = [c for c in X.columns if c.startswith('contribution')]
  A = [c for c in X.columns if c.startswith('numberOf') and c != 'numberOfHouses']
  a = X[A].to_numpy('float64')
  pc = X[[c for c in PCT_COLS if c in X.columns]].to_numpy('float64')
  F = {}
  F['agg_nProducts'] = (a > 0).sum(1)                 # distinct products held
  F['agg_nZeroPct'] = (pc == 0).sum(1)                # how narrow the zip profile is
  F['agg_nHighPct'] = (pc >= 5).sum(1)                # how many bands it dominates
  F['agg_nIncomeBands'] = (X[INCOME].to_numpy('float64') > 0).sum(1)
  F['agg_nEduBands'] = (X[EDU].to_numpy('float64') > 0).sum(1)
  for name, cols in SOCIO_GROUPS.items():             # spread of each ladder
    g = [c for c in cols if c in X.columns]
    F['agg_nb_' + name] = (X[g].to_numpy('float64') > 0).sum(1)
  F['agg_nPolicies'] = a.sum(1)
  F['agg_policiesPerProduct'] = F['agg_nPolicies'] / np.maximum(F['agg_nProducts'], 1)
  p = X[P].to_numpy('float64')
  for e in EXTRA_COUNTS:
    F['agg_' + e] = {'nContribTypes': lambda: (p > 0).sum(1),
                     'nVeryHighPct': lambda: (pc >= 8).sum(1),
                     'nMulti': lambda: (a >= 2).sum(1)}[e]()
  return {k: np.asarray(v, dtype='float64') for k, v in F.items()}


def _rare_count_cols(X):
  """numberOf* halves of the product pairs that are almost always zero."""
  out = []
  for c in X.columns:
    if c.startswith('contribution'):
      a = 'numberOf' + c[len('contribution'):]
      if a in X.columns and float((X[c].to_numpy('float64') != 0).mean()) < DROP_RARE_COUNTS:
        out.append(a)
  return out


def engineer(X_train, y_train, X_test):
  Ztr, Zte = X_train.copy(), X_test.copy()
  for k, v in _counts(X_train).items():   # assign, never concat: concat rebuilds string arrays
    Ztr[k] = v
  for k, v in _counts(X_test).items():
    Zte[k] = v
  drop = _rare_count_cols(X_train)
  return Ztr.drop(columns=drop), Zte.drop(columns=drop)


def postprocess(pred, y_orig, X_test):
  return pred
