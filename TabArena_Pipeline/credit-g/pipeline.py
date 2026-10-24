import numpy as np
import pandas as pd

TABFM_KWARGS = {'permute_categorical': True, 'n_svd_features': 'sqrt', 'n_feature_crosses': 'sqrt',
                'norm_methods': ['none', 'power', 'quantile', 'robust']}

# Risk orderings, ascending: the direction the Statlog attribute is defined in, and for the
# three-level columns the safest level first. credit_history's two middle levels are the one place
# this run overrules that reading (see the header).
ORD = {
    'checking_status': ['<0 DM', '0 <= ... < 200 DM',
                        '>= 200 DM / salary assignments for >= 1 year', 'no checking account'],
    'savings_status': ['< 100 DM', '100 <= ... < 500 DM', '500 <= ... < 1000 DM',
                       '>= 1000 DM', 'unknown / no savings'],
    'employment_since': ['unemployed', '< 1 year', '1 <= ... < 4 years',
                         '4 <= ... < 7 years', '>= 7 years'],
    'credit_history': ['no credits taken / all paid duly', 'all credits at this bank paid duly',
                       'delay in paying off in past', 'existing credits paid duly till now',
                       'critical account / other credits existing'],
    'property': ['real estate', 'building society savings / life insurance',
                 'car or other (not savings)', 'unknown / no property'],
    'other_debtors': ['guarantor', 'none', 'co-applicant'],
    'other_installment_plans': ['none', 'stores', 'bank'],
    'housing': ['rent', 'for free', 'own'],
    'telephone': ['none', 'yes, registered'],
    'foreign_worker': ['no', 'yes'],
}
RARE_FRAC = 35 / 599.0   # credit_purpose levels rarer than this share of a fold's training rows are
                         # folded into 'other'. It is 35 rows at the 599 a validation fold gives,
                         # which is what was tuned (20 costs +0.0022, 65 costs +0.0015), and it
                         # scales to ~39 at the 666 the real folds train on. That matters: the
                         # level it has to separate from 'business' (~68 rows there) is 'education',
                         # ~33 +- 5 rows, so a fixed 35 would leave education out of 'other' in
                         # roughly a third of the final's folds and silently do the worse thing.


def folds(X_train, y_train):
  from sklearn.model_selection import RepeatedStratifiedKFold
  return list(RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=0).split(X_train, y_train))


def preprocess(X_train, y_train, X_test):
  return X_train, y_train, X_test


def _ordinalise(X):
  """Ordered categoricals -> their rank on that ordering, in place."""
  X = X.copy()
  for c, order in ORD.items():
    if c in X.columns:
      X[c] = X[c].astype(str).map({v: float(i) for i, v in enumerate(order)})
  return X


def engineer(X_train, y_train, X_test):
  A, B = _ordinalise(X_train), _ordinalise(X_test)
  for f in (A, B):                                 # the two smallest male levels (35 and 57 rows)
    v = f['personal_status_sex'].to_numpy(dtype=object)      # say the same thing: not single
    f['personal_status_sex'] = np.array(['male: not single' if str(x).startswith('male') and
                                         'single' not in str(x) else x for x in v], dtype=object)
  for c in ('credit_purpose',):
    va, vb = A[c].to_numpy(dtype=object), B[c].to_numpy(dtype=object)
    vc = pd.Series(va).value_counts()
    rare = set(vc[vc < RARE_FRAC * len(va)].index)   # decided on the training rows, applied to both
    if rare:
      A[c] = np.array(['other' if v in rare else v for v in va], dtype=object)
      B[c] = np.array(['other' if v in rare else v for v in vb], dtype=object)
  for c in A.columns:                              # a string column built here comes out arrow-backed
    if not pd.api.types.is_numeric_dtype(A[c]):    # and the judge process cannot unpickle that;
      A[c], B[c] = A[c].astype(object), B[c].astype(object)      # plain object survives the trip
  return A, B


def postprocess(pred, y_orig, X_test):
  return pred
