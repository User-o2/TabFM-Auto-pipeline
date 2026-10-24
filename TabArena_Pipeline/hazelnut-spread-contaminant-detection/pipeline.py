import numpy as np
import pandas as pd

# intent: colorder rand4
TABFM_KWARGS = {
    # Eight members over five normalisations -- none, power and quantile get two each,
    # quantile_rtdl and robust one -- plus sqrt(30) random feature crosses on half of them.  The
    # order sets that split: moving robust forward, so it takes two members and quantile one, costs
    # +0.00014, and giving none or power a third member costs +0.00014 / +0.00009.
    'norm_methods': ['none', 'power', 'quantile', 'quantile_rtdl', 'robust'],
    'n_feature_crosses': 'sqrt',
    # 3.0 and the 4.0 default are indistinguishable (0.001517 vs 0.001521 over 50 splits) while
    'outlier_threshold': 3.0,
}


def preprocess(X_train, y_train, X_test):
  return X_train, y_train, X_test


COLORDER = 'rand4' 


def engineer(X_train, y_train, X_test):
  # No new columns -- but the frames still have to survive the trip home: the judge unpickles a
  # hook's output in an interpreter where importing pyarrow fails, and pandas 3 gives any index it
  # builds fresh a pyarrow-backed string dtype that cannot be loaded there.  Normalising the column
  # index to plain objects keeps that from biting a later edit that does build columns.
  if COLORDER != 'none':
    cols = [c for c in X_train.columns if c[0] == 's' and c[1:].isdigit()]
    rest = [c for c in X_train.columns if c not in cols]
    if COLORDER == 'rev':
      order = cols[::-1]
    else:
      rng = np.random.default_rng(int(COLORDER[4:]))
      order = [cols[i] for i in rng.permutation(len(cols))]
    X_train, X_test = X_train[rest + order], X_test[rest + order]
  return _plain(X_train), _plain(X_test)


def _plain(df):
  out = df.copy()
  out.columns = pd.Index(np.array([str(c) for c in df.columns], dtype=object), dtype=object)
  return out


def postprocess(pred, y_orig, X_test):
  return pred


def folds(X_train, y_train):
  """10-fold x 5 repeats.  The judge's default 3 folds spread by 4e-4 while the effects worth
  chasing here are 1e-4, and even 10 x 2 left a paired standard error of ~1.2e-4; 50 splits bring
  that to ~5e-5, and 1440 training rows per split also mirror the 1600 the final fits on.  The
  2400 rows hold no exact duplicates and no replicate clusters, so they are exchangeable and
  stratified k-fold is the right mirror of how the task's own folds are cut."""
  from sklearn.model_selection import RepeatedStratifiedKFold
  y = np.asarray(y_train).ravel()
  rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=1)
  return [(tr, va) for tr, va in rskf.split(np.zeros(len(y)), y)]
