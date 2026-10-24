import numpy as np

TABFM_KWARGS = {'norm_methods': ['power', 'quantile', 'quantile_rtdl', 'robust'],
                'outlier_threshold': 8.0}

DENS_KS = (2, 3, 6, 10, 20, 30, 60, 100, 200, 300)   # scales the density profile is sampled at

_METRO_PTS = np.array([(34.05, -118.24), (37.77, -122.42), (32.72, -117.16), (37.34, -121.89)])

# the shoreline of San Francisco / San Pablo / Suisun bay, anticlockwise from the Golden Gate.  Folded
# into the coastline rather than kept as its own column: as a column it lost, folded in it ties on
# score and puts Berkeley 3.6 km from the water instead of 25.
_BAY_PTS = np.array([
    (37.81, -122.48), (37.79, -122.39), (37.72, -122.38), (37.62, -122.36), (37.55, -122.27),
    (37.48, -122.20), (37.44, -122.10), (37.42, -121.98), (37.47, -121.94), (37.55, -122.03),
    (37.62, -122.10), (37.70, -122.17), (37.78, -122.25), (37.86, -122.31), (37.92, -122.35),
    (37.98, -122.40), (38.02, -122.28), (38.07, -122.25), (38.06, -122.13), (38.03, -122.03),
    (38.05, -121.90), (38.02, -121.80)])

# the California coastline, north to south, following the named headlands and bays
_COAST_PTS = np.array([
    (42.00, -124.21), (41.75, -124.20), (41.53, -124.08), (41.29, -124.10), (41.06, -124.14),
    (40.80, -124.16), (40.62, -124.31), (40.44, -124.41), (40.20, -124.35), (40.03, -124.07),
    (39.83, -123.90), (39.63, -123.78), (39.45, -123.81), (39.20, -123.77), (38.95, -123.74),
    (38.77, -123.53), (38.60, -123.35), (38.45, -123.13), (38.33, -123.05), (38.15, -122.97),
    (38.00, -123.02), (37.91, -122.68), (37.81, -122.48), (37.66, -122.50), (37.46, -122.44),
    (37.24, -122.42), (37.08, -122.27), (36.95, -122.03), (36.85, -121.81), (36.72, -121.80),
    (36.60, -121.89), (36.45, -121.92), (36.31, -121.89), (36.14, -121.65), (36.02, -121.55),
    (35.85, -121.42), (35.64, -121.19), (35.45, -120.94), (35.37, -120.86), (35.25, -120.75),
    (35.14, -120.64), (34.98, -120.65), (34.90, -120.67), (34.70, -120.60), (34.55, -120.65),
    (34.45, -120.47), (34.44, -120.05), (34.41, -119.69), (34.35, -119.47), (34.27, -119.29),
    (34.15, -119.21), (34.09, -119.06), (34.05, -118.85), (34.03, -118.68), (34.01, -118.50),
    (33.94, -118.43), (33.80, -118.42), (33.74, -118.40), (33.75, -118.19), (33.70, -118.08),
    (33.65, -118.00), (33.60, -117.90), (33.52, -117.78), (33.46, -117.70), (33.42, -117.62),
    (33.30, -117.50), (33.20, -117.39), (33.05, -117.31), (32.96, -117.27), (32.85, -117.27),
    (32.72, -117.25), (32.58, -117.13), (32.53, -117.12)])


def _densify(p, n=20):
  """polyline vertices plus interpolated points, so a min over points is a min over the line"""
  t = np.linspace(0.0, 1.0, n, endpoint=False)[:, None, None]
  return np.vstack([(p[:-1][None] * (1 - t) + p[1:][None] * t).reshape(-1, 2), p[-1:]])


def _min_dist(lat, lon, c, chunk=2048):
  """km to the nearest of a set of reference points, in row blocks: the coastline is 1862 points and
  the whole-fold cross product would be a 200 MB temporary for no reason"""
  out = np.empty(len(lat))
  for i in range(0, len(lat), chunk):
    la, lo = lat[i:i + chunk], lon[i:i + chunk]
    dlat = la[:, None] - c[None, :, 0]
    dlon = (lo[:, None] - c[None, :, 1]) * np.cos(np.deg2rad(la))[:, None]
    out[i:i + chunk] = np.sqrt(dlat ** 2 + dlon ** 2).min(1)
  return 111.0 * out


def _density(Xtr, Xte, ks):
  """km to the k-th nearest block group, self excluded, over the pooled train+test coordinates.

  Pooling the test coordinates -- their features, never their targets -- is worth -0.0006: half again
  as many points to estimate from, and one estimator instead of two subtly different ones.  Weighting
  the neighbours by their household count is worse: census block groups are already drawn to hold
  similar populations, so counting them is the population density and weighting adds boundary noise."""
  from sklearn.neighbors import KDTree
  s = np.array([1.0, np.cos(np.deg2rad(35.6))])
  P = np.c_[Xtr.Latitude.to_numpy(float), Xtr.Longitude.to_numpy(float)] * s
  Q = np.c_[Xte.Latitude.to_numpy(float), Xte.Longitude.to_numpy(float)] * s
  R = np.vstack([P, Q])
  ks = [min(k, len(R) - 1) for k in ks]
  kmax = max(ks)
  d, i = KDTree(R).query(R, k=kmax + 1)
  selfm = i == np.arange(len(R))[:, None]   # co-located blocks tie at distance 0: self is not always column 0
  selfm[~selfm.any(1), -1] = True
  d = d[~selfm].reshape(len(d), kmax)
  return {f'dens{k}_km': (111.0 * d[:len(P), k - 1], 111.0 * d[len(P):, k - 1]) for k in ks}


def preprocess(X_train, y_train, X_test):
  return X_train, y_train, X_test


def _rewrite(d):
  """the four collinear size columns as one level plus three ratios, plus the two geographic minima"""
  o = d.drop(columns=['TotalRooms', 'TotalBedrooms', 'Population'])
  o['rooms_per_hh'] = d.TotalRooms / d.Households      # dwelling size
  o['bed_per_room'] = d.TotalBedrooms / d.TotalRooms   # dwelling type, orthogonal to its size
  o['pop_per_hh'] = d.Population / d.Households        # household size
  la, lo = d.Latitude.to_numpy(float), d.Longitude.to_numpy(float)
  o['coast_km'] = _min_dist(la, lo, np.vstack([_densify(_COAST_PTS), _densify(_BAY_PTS)]))
  m = np.sort(np.stack([_min_dist(la, lo, _METRO_PTS[j:j + 1]) for j in range(len(_METRO_PTS))]), axis=0)
  for j in range(len(_METRO_PTS)):                       # the four expensive coastal markets, sorted:
    o[f'metro{j + 1}_km'] = m[j]                         # metro1_km is the old min column
  return o


def engineer(X_train, y_train, X_test):
  A, B = _rewrite(X_train), _rewrite(X_test)
  for name, (a, b) in _density(X_train, X_test, DENS_KS).items():
    A[name], B[name] = a, b
  return A, B[A.columns]


def postprocess(pred, y_orig, X_test):
  y = np.asarray(y_orig, dtype=float).ravel()          # the target is censored: 13.1224 is ln(500001)
  return np.clip(pred, y.min(), y.max())               # and 4.7% of the rows sit exactly on it
