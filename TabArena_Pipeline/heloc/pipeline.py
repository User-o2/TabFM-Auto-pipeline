import numpy as np

# One view, the strongest one, on all eight members; the column permutation is the only
# randomisation left and it is the only one that earns its keep.
TABFM_KWARGS = {"norm_methods": ["quantile"], "class_shift": False}

SORT_CONTEXT = "ExternalRiskEstimate"


def folds(X_train, y_train):
  """3x repeated stratified 5-fold. Three folds left a 0.0035 spread between folds, wide
  enough to hide everything measured here; repeats average the partition away, and a
  5-fold context (5578 rows) is also much closer to the 6972 the final fits on."""
  from sklearn.model_selection import RepeatedStratifiedKFold
  y = y_train.iloc[:, 0] if hasattr(y_train, "columns") else y_train
  rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0)
  return list(rskf.split(np.zeros(len(y)), y))


def preprocess(X_train, y_train, X_test):
  return X_train, y_train, X_test


def _add(X):
  """The 23 raw columns untouched, plus one re-ordered copy of the one column FICO orders
  backwards. Only the order of these four values matters -- the members rank every column
  -- so: -8 ("no inquiries at all") above every real month count, -7 ("an inquiry inside
  the last 7 days") just below zero, -9 ("no bureau record") at the bottom."""
  X = X.copy()
  q = X["MSinceMostRecentInqexcl7days"]
  X["inq_mono"] = (q.where(q >= 0)
                   .mask(q == -8, 40.0)
                   .mask(q == -7, -1.0)
                   .mask(q == -9, -2.0))
  return X


def engineer(X_train, y_train, X_test):
  return _add(X_train), _add(X_test)


def sample(X_train, y_train, X_test, max_rows):
  """One draw of every training row: every variant that removed rows lost, roughly in
  proportion to how many, and no amount of extra ensembling bought it back."""
  return [np.argsort(X_train[SORT_CONTEXT].to_numpy(), kind="stable")]


def postprocess(pred, y_orig, X_test):
  return pred
