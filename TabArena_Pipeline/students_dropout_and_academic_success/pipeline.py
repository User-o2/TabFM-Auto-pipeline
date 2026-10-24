import re

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

TABFM_KWARGS = {
    'multiclass_calibration_method': 'vector',
    'calibration_lambda': 0.001,
    'average_logits': False,     # required by enable_nnls
    'enable_nnls': True,
    'nnls_beta': 1.0,
    'permute_categorical': True,
    'feat_shuffle_method': 'none',
}

# (regex, years-of-schooling equivalent); first match wins, so the specific rules come first
EDU_RULES = [
    (r"doctorate", 21),
    (r"master", 18),
    (r"specialized higher studies", 17),
    (r"bachelor|higher education - degree|degree \(1st cycle\)", 16),
    (r"professional higher technical", 14),
    (r"frequency of higher education", 13),
    (r"technological specialization", 13),
    (r"short cycle diploma", 13),
    (r"technical-professional", 12),
    (r"secondary education|12th year of schooling or eq", 12),
    (r"12th year of schooling - not completed", 11),
    (r"general commerce|2nd cycle of the general high school", 11),
    (r"other - 11th year", 11),
    (r"complementary high school course - not concluded", 10),
    (r"complementary high school", 11),
    (r"general course of administration and commerce|supplementary accounting", 11),
    (r"11th year of schooling - not completed", 10),
    (r"10th year of schooling - not completed", 9),
    (r"10th year of schooling", 10),
    (r"9th year of schooling - not completed", 8),
    (r"3rd cycle", 9),
    (r"8th year of schooling", 8),
    (r"7th year", 7),
    (r"2nd cycle", 6),
    (r"1st cycle \(4th/5th year\)|basic education 1st cycle", 4),
    (r"can read without", 2),
    (r"can't read or write|cannot read", 0),
]

# (regex, ISCO-style skill rank, 9 = managers ... 2 = elementary occupations)
OCC_RULES = [
    (r"representatives of legislative|directors of", 9),
    (r"specialists in intellectual|health professionals|teachers|finance/administration specialists"
     r"|ict specialists", 8),
    (r"intermediate level technicians|technicians|sergeants|finance/statistics", 7),
    (r"administrative staff|administrative support|office workers", 6),
    (r"personal service|personal care|sellers|meal preparation|armed forces", 5),
    (r"unskilled|subsistence|cleaning workers", 2),
    (r"skilled construction|skilled workers|metal workers|electricians|food/craft|craftsmen"
     r"|food/woodworking", 4),
    (r"farmers", 4),
    (r"machine operators|installation and machine|assembly workers|vehicle drivers", 3),
    (r"student", None),          # a parent still studying is not a rung on this ladder
]

# source column -> the rank column added beside it. 'Unknown', '(blank)' and 'Other Situation'
# match no rule and stay NaN, which TabFM handles natively.
RANKED = [("Mothers_qualification", "mother_edu", EDU_RULES),
          ("Fathers_qualification", "father_edu", EDU_RULES),
          ("Mothers_occupation", "mother_occ", OCC_RULES),
          ("Fathers_occupation", "father_occ", OCC_RULES)]


def _rank_column(values, rules):
  """Position on the ladder, evenly spaced. The ladder comes from the rule table rather than from
  the rows, so every fold -- and the held-out split -- shares one scale."""
  ladder = {v: float(i) for i, v in enumerate(sorted({r for _, r in rules if r is not None}))}
  cache = {}
  out = np.empty(len(values))
  for i, v in enumerate(values):
    if v not in cache:
      s = str(v).lower()
      hit = next((r for pat, r in rules if re.search(pat, s)), None)
      cache[v] = ladder[hit] if hit is not None else np.nan
    out[i] = cache[v]
  return out


def folds(X_train, y_train):
  """2 repeats of stratified 4-fold: every row scored twice, on a 2212-row context."""
  y = np.asarray(y_train).ravel()
  cv = RepeatedStratifiedKFold(n_splits=4, n_repeats=2, random_state=20260907)
  return [(tr, va) for tr, va in cv.split(np.zeros(len(y)), y)]


def preprocess(X_train, y_train, X_test):
  return X_train, y_train, X_test


def engineer(X_train, y_train, X_test):
  out = []
  for X in (X_train, X_test):
    X = X.copy()
    for src, dst, rules in RANKED:
      if src in X.columns:
        X[dst] = _rank_column(np.asarray(X[src], dtype=object), rules)
    out.append(X)
  return out[0], out[1]


def postprocess(pred, y_orig, X_test):
  return pred
