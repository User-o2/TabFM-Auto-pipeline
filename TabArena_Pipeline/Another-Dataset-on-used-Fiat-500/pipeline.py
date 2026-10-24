import numpy as np

TABFM_KWARGS = {}

W_MAX = 0.55      # blend weight at full distance and full mileage
D0 = 1.0          # distance scale, in units of the context's own median nearest-neighbour spacing
KMF = 2.0         # mileage scale, in units of the fold's median km
BSTASH = '.blend.npy'
WSTASH = '.wblend.npy'


def folds(X_train, y_train):
  """3 repeats of 5-fold: every row is scored under three different partitions, which is what makes a
  2-point difference readable on 1025 rows. At a single 5-fold the whole field sat inside 720-724."""
  from sklearn.model_selection import RepeatedKFold
  return list(RepeatedKFold(n_splits=5, n_repeats=3, random_state=0).split(X_train))


def _design(X, cols=None):
  """Numeric view of the table for the helper models; the id column is not a feature."""
  import pandas as pd
  A = pd.get_dummies(X.drop(columns=[c for c in ['row_id'] if c in X.columns]),
                     columns=['model']).astype(float)
  return A if cols is None else A.reindex(columns=cols, fill_value=0.0)


def _partner(A, y, B):
  """The second opinion, fitted on the fold's training rows only."""
  from sklearn.ensemble import GradientBoostingRegressor
  g = GradientBoostingRegressor(random_state=0, n_estimators=600, learning_rate=0.05,
                                max_depth=2, min_samples_leaf=5)
  return g.fit(A, y).predict(B)


def _weights(A, B):
  """How much of the query row to hand to the partner. Near a context row TabFM's in-context fit is
  close to the local noise floor and averaging only adds error; in empty space both models are
  guessing and averaging pays. Both scales are read off the training rows, so nothing is hard-coded
  to this fold's size."""
  from sklearn.neighbors import NearestNeighbors
  from sklearn.preprocessing import QuantileTransformer
  qt = QuantileTransformer(n_quantiles=min(300, len(A)), output_distribution='normal').fit(A)
  Zt = qt.transform(A)
  nn = NearestNeighbors(n_neighbors=2).fit(Zt)
  dref = max(float(np.median(nn.kneighbors(Zt)[0][:, 1])), 1e-9)
  d = nn.kneighbors(qt.transform(B))[0][:, 0]
  kmed = max(float(np.median(A['km'].to_numpy())), 1.0)
  return (W_MAX * np.clip(d / (D0 * dref), 0.0, 1.0)
          * np.clip(B['km'].to_numpy() / (KMF * kmed), 0.0, 1.0))


def preprocess(X_train, y_train, X_test):
  """TabFM and the partner both work on sqrt(price); postprocess squares the blend back."""
  import pandas as pd
  A, B = _design(X_train), _design(X_test, _design(X_train).columns)
  z = np.sqrt(np.asarray(y_train, dtype=float))
  np.save(BSTASH, _partner(A.to_numpy(), z, B.to_numpy()))
  np.save(WSTASH, _weights(A, B))
  return X_train, pd.Series(z, index=y_train.index, name=y_train.name), X_test


def engineer(X_train, y_train, X_test):
  return X_train, X_test


def postprocess(pred, y_orig, X_test):
  import pandas as pd
  w = np.load(WSTASH)
  z = (1 - w) * np.asarray(pred, dtype=float) + w * np.load(BSTASH)
  return pd.Series(np.square(z), index=X_test.index)
