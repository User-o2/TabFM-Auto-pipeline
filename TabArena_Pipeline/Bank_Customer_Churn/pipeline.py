import numpy as np
import pandas as pd

# class_shift and the normaliser list are the only two constructor knobs that beat the defaults;
TABFM_KWARGS = {'class_shift': False, 'norm_methods': ['none']}

DROP = ('estimated_salary', 'credit_card', 'tenure')
AS_LEVELS = ('products_number', 'active_member')


def folds(X_train, y_train):
  """5x repeated stratified 3-fold. The judge's default 3-fold spread is std=0.009 on a 0.126
  score, which cannot rank changes worth 0.0005; the repeats average the split noise out while
  keeping the 2:1 train/validation ratio of the TabArena folds the final is scored on."""
  from sklearn.model_selection import RepeatedStratifiedKFold
  y = np.asarray(y_train).ravel()
  cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=5, random_state=0)
  return list(cv.split(np.zeros(len(y)), y))


def preprocess(X_train, y_train, X_test):
  return X_train, y_train, X_test


def _build(df):
  """Every column here is a function of that row alone, so train and test need no shared state."""
  d = df.drop(columns=[c for c in DROP if c in df.columns])
  for c in AS_LEVELS:
    d[c] = _object([('v%d' % v) for v in df[c].astype(int)], d.index)
  d['bal_zero'] = (df['balance'] == 0).astype(int)
  d['balance'] = df['balance'].where(df['balance'] > 0, np.nan)
  return d


def _object(values, index):
  """Force plain-numpy object dtype.

  The hooks run in a pandas 3.0 that has pyarrow, so a string column assigned normally becomes an
  ArrowStringArray; the judge unpickles the result in a pandas 3.0 that does not have pyarrow and
  refuses to load one. `judge.py check` runs in-process and never crosses that boundary, so it
  reports ok either way -- this cost three evaluations to find.
  """
  return pd.Series(np.asarray(values, dtype=object), index=index, dtype=object)


def engineer(X_train, y_train, X_test):
  a, b = _build(X_train), _build(X_test)
  for d in (a, b):
    for c in d.columns:
      if not pd.api.types.is_numeric_dtype(d[c]):
        d[c] = _object(d[c], d.index)
  return a, b


def postprocess(pred, y_orig, X_test):
  return pred
