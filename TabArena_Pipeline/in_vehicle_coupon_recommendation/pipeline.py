import numpy as np
import pandas as pd

TABFM_KWARGS = {'n_feature_crosses': 'sqrt', 'n_svd_features': 'sqrt',
                'norm_methods': ['none', 'power', 'quantile', 'quantile_rtdl', 'robust'],
                'permute_categorical': True}

FREQ = {'never': 0.0, 'less1': 1.0, '1~3': 2.0, '4~8': 3.0, 'gt8': 4.0}
FREQ_COLS = ['Bar', 'CoffeeHouse', 'CarryAway', 'RestaurantLessThan20', 'Restaurant20To50']
AGE = {'below21': 18.0, '21': 21.0, '26': 26.0, '31': 31.0, '36': 36.0, '41': 41.0,
       '46': 46.0, '50plus': 55.0}
INCOME = {'Less than $12500': 6.25, '$12500 - $24999': 18.75, '$25000 - $37499': 31.25,
          '$37500 - $49999': 43.75, '$50000 - $62499': 56.25, '$62500 - $74999': 68.75,
          '$75000 - $87499': 81.25, '$87500 - $99999': 93.75, '$100000 or More': 112.5}
EDU = {'Some High School': 0.0, 'High School Graduate': 1.0, 'Some college - no degree': 2.0,
       'Associates degree': 3.0, 'Bachelors degree': 4.0,
       'Graduate degree (Masters or Doctorate)': 5.0}
EXPIRY = {'2h': 2.0, '1d': 24.0}

# which visit-frequency column the coupon is asking about
COUPON_VENUE = {'Bar': 'Bar', 'Coffee House': 'CoffeeHouse',
                'Carry out & Take away': 'CarryAway',
                'Restaurant(<20)': 'RestaurantLessThan20',
                'Restaurant(20-50)': 'Restaurant20To50'}

# direction_opp is 1 - direction_same; car is ~99% missing; the two flags become `distance`
DROP = ['direction_opp', 'car', 'toCoupon_GEQ15min', 'toCoupon_GEQ25min']

PAIRS = [('coupon', 'destination'), ('coupon', 'passenger'), ('coupon', 'time'), ('coupon', 'expiration'), ('coupon', 'CoffeeHouse'), ('coupon', 'income')]


def _map(s, table):
  return s.astype('object').map(table).astype(float)


def preprocess(X_train, y_train, X_test):
  return X_train, y_train, X_test


def _recode(X):
  X = X.copy()
  nmiss = X[FREQ_COLS].isna().sum(axis=1).astype(float)
  for c in FREQ_COLS:
    X[c] = _map(X[c], FREQ)
  X['age'] = _map(X['age'], AGE)
  X['income'] = _map(X['income'], INCOME)
  X['education'] = _map(X['education'], EDU)
  X['expiration'] = _map(X['expiration'], EXPIRY)
  # the two flags are nested (>=25min implies >=15min): one ordered level instead
  X['distance'] = X['toCoupon_GEQ15min'].astype(float) + X['toCoupon_GEQ25min'].astype(float)

  coup = X['coupon'].astype('object')            # how often this driver visits the
  own = np.full(len(X), np.nan)                  # venue type the coupon is for
  for c, col in COUPON_VENUE.items():
    m = (coup == c).to_numpy()
    own[m] = X.loc[m, col].to_numpy()
  X['venue_freq'] = own
  X['out_freq'] = X[FREQ_COLS].sum(axis=1, min_count=1).astype(float)
  X['venue_share'] = X['venue_freq'] / (X['out_freq'] + 1.0)

  # a numeric column is mean-imputed by TabFM, which would erase the blanks the bands had
  for c in FREQ_COLS:
    X[c] = X[c].fillna(-1.0)
  X['n_missing'] = nmiss

  for a, b in PAIRS:
    # plain object dtype: an Arrow-backed result does not survive the hook pickle
    va, vb = X[a].astype('object').to_numpy(), X[b].astype('object').to_numpy()
    X[f'{a}_x_{b}'] = pd.Series([f'{u}|{v}' for u, v in zip(va, vb)],
                                index=X.index, dtype=object)

  return X.drop(columns=[c for c in DROP if c in X.columns])


def engineer(X_train, y_train, X_test):
  return _recode(X_train), _recode(X_test)


def postprocess(pred, y_orig, X_test):
  return pred


def folds(X_train, y_train):
  """5 folds instead of the judge's 3: paired comparisons between two pipelines are what
  this search reads, and 5 folds both halve the noise on that difference and put the
  per-fold training size (6765) nearer the 11415 rows the final actually fits on."""
  from sklearn.model_selection import StratifiedKFold
  return list(StratifiedKFold(5, shuffle=True, random_state=0).split(X_train, y_train))
